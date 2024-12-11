import torch
from torch import nn
from torch.nn import functional as F


class PatchNCELoss(nn.Module):
    """
    Patch-wise NCE loss for Swapping Autoencoder
    
    Args:
        nce_T (float): temperature for NCE loss
        batch (int): batch size
    """
    def __init__(self, nce_T, batch):
        super().__init__()
        self.nce_T = nce_T
        self.batch = batch
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit
        batch_dim_for_bmm = self.batch
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[
            None, :, :
        ]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        return self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real_list = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = (
        grad_real_list[0].pow(2).reshape(grad_real_list[0].shape[0], -1).sum(1).mean()
    )
    if len(grad_real_list) > 1:
        for grad_real in grad_real_list[1:]:
            grad_penalty += (
                grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            )

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()
