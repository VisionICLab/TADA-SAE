import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
from training.trainers import Trainer
from models.swapping_autoencoder.loss import (
    d_logistic_loss,
    g_nonsaturating_loss,
    PatchNCELoss,
    d_r1_loss,
)
from models.swapping_autoencoder import utils
from inference.pipelines.sae import generate_interpolation


REF_CROP = 4
N_CROPS = 8
D_REG_EVERY = 16


class SAETrainer(Trainer):
    def __init__(
        self,
        encoder,
        generator,
        str_projector,
        discriminator,
        cooccur_disc,
        enc_ema,
        gen_ema,
        ada_aug,
        config,
        logger,
        scheduler=None,
    ):
        super().__init__(None, None, None, config, logger, scheduler)
        self.encoder = encoder
        self.generator = generator
        self.str_projector = str_projector
        self.discriminator = discriminator
        self.cooccur_disc = cooccur_disc

        self.patchnce_loss = PatchNCELoss(
            nce_T=0.07, batch=self.config["batch_size"] // 2
        ).to(self.config["device"])

        self.patchify_image = partial(
            utils.raw_patchify_image,
            min_size=self.config["min_patch_size"],
            max_size=self.config["max_patch_size"],
        )

        self.g_optim = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.generator.parameters())
            + list(self.str_projector.parameters()),
            lr=self.config["lr"],
            betas=(0.5, 0.99),
        )

        self.d_optim = torch.optim.Adam(
            list(self.discriminator.parameters())
            + list(self.cooccur_disc.parameters()),
            lr=self.config["lr"],
            betas=(0.5, 0.99),
        )

        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.g_optim, gamma=0.99
        )
        self.disc_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.d_optim, gamma=0.99
        )

        self.g_grad_scaler = GradScaler()
        self.d_grad_scaler = GradScaler()

        self.enc_ema = enc_ema
        self.gen_ema = gen_ema
        self.ada_aug = ada_aug
        self.global_step = 0

    def _r1_d_regularize(self, real_img, real_patch, ref_patch):
        R1 = 10
        COOCUR_R1 = 1
        real_img.requires_grad = True
        self.d_optim.zero_grad()
        with autocast():
            real_pred = self.discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            real_patch.requires_grad = True
            real_patch_pred, _ = self.cooccur_disc(
                real_patch, N_CROPS, reference=ref_patch, ref_batch=REF_CROP
            )
            cooccur_r1_loss = d_r1_loss(real_patch_pred, real_patch)
            r1_loss_sum = (
                R1 / 2 * r1_loss * D_REG_EVERY
                + COOCUR_R1 / 2 * cooccur_r1_loss * D_REG_EVERY
            )

        self.d_grad_scaler.scale(r1_loss_sum).backward()
        self.d_grad_scaler.unscale_(self.d_optim)
        torch.nn.utils.clip_grad_norm_(
            list(self.discriminator.parameters())
            + list(self.cooccur_disc.parameters()),
            self.config["grad_clip"],
        )
        self.d_grad_scaler.step(self.d_optim)
        self.d_grad_scaler.update()

    def feat_recons_loss(self, fake_struct1, struct1, fake_tex2, tex2):
        return (F.mse_loss(fake_struct1, struct1) + F.mse_loss(fake_tex2, tex2)) / 2

    def pnce_loss(self, str_qs, str_ks):
        g_pnce_loss = 0
        num_scales = len(str_qs)
        for str_q, str_k in zip(str_qs, str_ks):
            g_pnce_loss += self.patchnce_loss(str_q, str_k) / num_scales
        return g_pnce_loss

    def train_step(self, im, mask):
        real_img, real_mask = im.to(self.config["device"]), mask.to(
            self.config["device"]
        )
        real_img1, real_img2 = real_img.chunk(2, dim=0)
        real_mask1, real_mask2 = real_mask.chunk(2, dim=0)

        utils.requires_grad(self.encoder, False)
        utils.requires_grad(self.generator, False)
        utils.requires_grad(self.str_projector, False)
        utils.requires_grad(self.discriminator, True)
        utils.requires_grad(self.cooccur_disc, True)

        real_img_aug, _ = self.ada_aug(real_img, real_mask)

        with autocast():
            structure1, texture1 = self.encoder(real_img1, multi_tex=False)
            _, texture2 = self.encoder(real_img2, run_str=False, multi_tex=False)

            # image adversarial loss
            fake_img1 = self.generator(structure1, texture1)
            fake_img2 = self.generator(structure1, texture2)

            # augmentations following StyleGAN2-ADA
            fake_img = torch.cat((fake_img1, fake_img2), 0)
            fake_img_aug, _ = self.ada_aug(fake_img)
            fake_pred = self.discriminator(fake_img_aug)
            real_pred = self.discriminator(real_img_aug)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            # texture adv loss
            fake_patch, _ = self.patchify_image(fake_img2, N_CROPS, mask=real_mask1)
            real_patch, _ = self.patchify_image(real_img2, N_CROPS, mask=real_mask2)
            ref_patch, _ = self.patchify_image(
                real_img2, REF_CROP * N_CROPS, mask=real_mask2
            )
            fake_patch_pred, ref_input = self.cooccur_disc(
                fake_patch, N_CROPS, reference=ref_patch, ref_batch=REF_CROP
            )
            real_patch_pred, _ = self.cooccur_disc(
                real_patch, N_CROPS, ref_input=ref_input
            )
            d_cooccur_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)

        self.d_optim.zero_grad()
        d_total_loss = d_loss + d_cooccur_loss
        self.d_grad_scaler.scale(d_total_loss).backward()
        self.d_grad_scaler.unscale_(self.d_optim)
        torch.nn.utils.clip_grad_norm_(
            list(self.discriminator.parameters())
            + list(self.cooccur_disc.parameters()),
            self.config["grad_clip"],
        )
        self.d_grad_scaler.step(self.d_optim)
        self.d_grad_scaler.update()

        if self.global_step % D_REG_EVERY == 0:
            with autocast():
                real_patch, _ = self.patchify_image(real_img2, N_CROPS, mask=real_mask2)
                ref_patch, _ = self.patchify_image(
                    real_img2, REF_CROP * N_CROPS, mask=real_mask2
                )
            self._r1_d_regularize(real_img, real_patch, ref_patch)

        utils.requires_grad(self.encoder, True)
        utils.requires_grad(self.generator, True)
        utils.requires_grad(self.str_projector, True)
        utils.requires_grad(self.discriminator, False)
        utils.requires_grad(self.cooccur_disc, False)

        with autocast():
            structure1_list, texture1 = self.encoder(
                real_img1, multi_str=True, multi_tex=False
            )
            _, texture2 = self.encoder(real_img2, run_str=False, multi_tex=False)
            structure1 = structure1_list[-1]

            fake_img1 = self.generator(structure1, texture1)
            fake_img2 = self.generator(structure1, texture2)
            recon_loss = F.l1_loss(fake_img1, real_img1.detach())

            # image adversarial loss (for generator)
            fake_img = torch.cat((fake_img1, fake_img2), 0)
            fake_img_aug, _ = self.ada_aug(fake_img)
            fake_pred = self.discriminator(fake_img_aug)
            g_loss = g_nonsaturating_loss(fake_pred)

            # texture adversarial loss (for generator)
            fake_patch, _ = self.patchify_image(fake_img2, N_CROPS, real_mask1)
            ref_patch, _ = self.patchify_image(
                real_img2, REF_CROP * N_CROPS, real_mask2
            )
            fake_patch_pred, _ = self.cooccur_disc(
                fake_patch, N_CROPS, reference=ref_patch, ref_batch=REF_CROP
            )
            g_cooccur_loss = g_nonsaturating_loss(fake_patch_pred)

            # Patch NCE loss
            # re-encode
            fake_structure1_list, fake_texture2 = self.encoder(
                fake_img2, multi_str=True, multi_tex=False
            )
            fake_patch_vectors, coords = utils.sample_patches(
                fake_structure1_list[:-1], N_CROPS, mask=real_mask1, inv=True
            )
            real_patch_vectors, _ = utils.sample_patches(
                structure1_list[:-1], N_CROPS, coords=coords
            )
            str_qs = self.str_projector(fake_patch_vectors)
            str_ks = self.str_projector(real_patch_vectors)

            g_pnce_loss = self.pnce_loss(str_qs, str_ks)

            # feature reconstruction loss
            feat_recon_loss = self.feat_recons_loss(
                fake_structure1_list[-1],
                structure1.detach(),
                fake_texture2,
                texture2.detach(),
            )
            gen_total_loss = (
                recon_loss + 0.5*g_loss + g_pnce_loss + g_cooccur_loss + 0.5*feat_recon_loss
            )

        self.g_optim.zero_grad()
        self.g_grad_scaler.scale(gen_total_loss).backward()
        self.g_grad_scaler.unscale_(self.g_optim)
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.generator.parameters())
            + list(self.str_projector.parameters()),
            self.config["grad_clip"],
        )
        self.g_grad_scaler.step(self.g_optim)
        self.g_grad_scaler.update()

        self.enc_ema.update()
        self.gen_ema.update()
        self.ada_aug.step(real_pred)

        self.logger.register_log(
            {
                "d_loss": d_loss.item(),
                "d_cooccur_loss": d_cooccur_loss.item(),
                "d_total_loss": d_total_loss.item(),
                "recon_loss": recon_loss.item(),
                "g_loss": g_loss.item(),
                "g_cooccur_loss": g_cooccur_loss.item(),
                "g_pnce_loss": g_pnce_loss.item(),
                "feat_recon_loss": feat_recon_loss.item(),
                "g_total_loss": gen_total_loss.item(),
                "aug_p": self.ada_aug.aug_p,
            }
        )

    @torch.no_grad()
    def eval_step(self, im, mask):
        real_img = im.to(self.config["device"])
        real_mask = mask.to(self.config["device"])

        real_img1, real_img2 = real_img.chunk(2, dim=0)
        real_mask1, _ = real_mask.chunk(2, dim=0)

        structure1_list, texture1 = self.encoder(
            real_img1, multi_str=True, multi_tex=False
        )
        structure1 = structure1_list[-1]
        _, texture2 = self.encoder(real_img2, run_str=False, multi_tex=False)

        fake_img1 = self.generator(structure1, texture1)
        fake_img2 = self.generator(structure1, texture2)

        fake_structure1_list, fake_texture2 = self.encoder(
            fake_img2, multi_str=True, multi_tex=False
        )
        fake_patch_vectors, coords = utils.sample_patches(
            fake_structure1_list[:-1], N_CROPS, mask=real_mask1, inv=True
        )
        real_patch_vectors, _ = utils.sample_patches(
            structure1_list[:-1], N_CROPS, coords=coords
        )
        str_qs = self.str_projector(fake_patch_vectors)
        str_ks = self.str_projector(real_patch_vectors)

        g_pnce_loss = self.pnce_loss(str_qs, str_ks)
        recons_loss = F.l1_loss(fake_img1, real_img1)
        feat_recon_loss = self.feat_recons_loss(
            fake_structure1_list[-1],
            structure1.detach(),
            fake_texture2,
            texture2.detach(),
        )

        self.logger.register_log(
            {
                "recons_loss_eval": recons_loss.item(),
                "g_pnce_loss_eval": g_pnce_loss.item(),
                "feat_recon_loss_eval": feat_recon_loss.item(),
            }
        )

    def train(self, train_loader):
        self.encoder.train()
        self.str_projector.train()
        self.generator.train()
        self.discriminator.train()
        self.cooccur_disc.train()

        progress_bar = tqdm(
            range(self.config["log_each"]), desc="Training", colour="green"
        )
        for _ in progress_bar:
            im, mask = next(train_loader)
            self.train_step(im, mask)
            self.global_step += 1

    @torch.no_grad()
    def evaluate(self, val_loader, train_loader):
        self.encoder.eval()
        self.generator.eval()

        progress_bar = tqdm(val_loader, desc="Evaluating reconstruction", colour="cyan")
        for im, mask in progress_bar:
            self.eval_step(im, mask)

        im, _ = next(iter(val_loader))
        f = self._make_visualisations(im, "Interpolation - Eval")
        self.logger.register_visualization("interpolation_eval", f)

        im, _ = next(iter(train_loader))
        f = self._make_visualisations(im, "Interpolation - Train")
        self.logger.register_visualization("interpolation_train", f)

    def save_state(self, path):
        if torch.cuda.device_count() > 1:
            encoder = self.encoder.module
            generator = self.generator.module
            str_projector = self.str_projector.module
            discriminator = self.discriminator.module
            cooccur_disc = self.cooccur_disc.module
        else:
            encoder = self.encoder
            generator = self.generator
            str_projector = self.str_projector
            discriminator = self.discriminator
            cooccur_disc = self.cooccur_disc

        torch.save(
            {
                "encoder": encoder.state_dict(),
                "generator": generator.state_dict(),
                "enc_ema": self.enc_ema.state_dict(),
                "gen_ema": self.gen_ema.state_dict(),
                "str_projector": str_projector.state_dict(),
                "discriminator": discriminator.state_dict(),
                "cooccur_disc": cooccur_disc.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "scheduler": (
                    None if self.scheduler is None else self.scheduler.state_dict()
                ),
                "global_step": self.global_step,
                "ada_aug_p": self.ada_aug.aug_p,
            },
            path,
        )

    def load_state(self, path):
        checkpoint = torch.load(path, map_location=self.config["device"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.generator.load_state_dict(checkpoint["generator"])
        self.enc_ema.load_state_dict(checkpoint["enc_ema"])
        self.gen_ema.load_state_dict(checkpoint["gen_ema"])
        self.str_projector.load_state_dict(checkpoint["str_projector"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.cooccur_disc.load_state_dict(checkpoint["cooccur_disc"])
        self.g_optim.load_state_dict(checkpoint["g_optim"])
        self.d_optim.load_state_dict(checkpoint["d_optim"])
        self.ada_aug.aug_p = checkpoint["ada_aug_p"]

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint["global_step"]

        if torch.cuda.device_count() > 1:
            self.encoder = DataParallel(self.encoder)
            self.generator = DataParallel(self.generator)
            self.str_projector = DataParallel(self.str_projector)
            self.discriminator = DataParallel(self.discriminator)
            self.cooccur_disc = DataParallel(self.cooccur_disc)

    def fit(self, train_loader, val_loader):
        while self.global_step < self.config["steps"]:
            self.train(train_loader)
            self.evaluate(val_loader, train_loader)
            self.logger.compile_logs()
            self.g_scheduler.step()
            self.disc_scheduler.step()
            self.logger.log(self.global_step)
            self._make_checkpoint(self.global_step)

        checkpoint_path = os.path.join(
            self.config["log_dir"], "checkpoints", self.config["run_name"]
        )
        self.save_state(f"{checkpoint_path}_{self.global_step}_final.pth")

    @torch.no_grad()
    def _make_visualisations(self, im_batch, title):
        struct_img = im_batch[0].unsqueeze(0).to(self.config["device"])
        text_img = im_batch[1].unsqueeze(0).to(self.config["device"])
        (recons_img1, recons_img2), (interp_img1, interp_img2) = generate_interpolation(
            self.enc_ema, self.gen_ema, struct_img, text_img
        )

        imgs = torch.cat(
            (struct_img, text_img, recons_img1, recons_img2, interp_img1, interp_img2),
            dim=0,
        )
        grid = make_grid(imgs, nrow=2, normalize=True).permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots(num=1, clear=True)
        fig.tight_layout()
        ax.set_axis_off()
        ax.set_title(title)

        ax.imshow(grid, vmin=0, vmax=1)
        return ax.figure
