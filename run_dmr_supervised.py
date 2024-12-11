import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from datasets.dmrir_dataset import DMRIRLeftRightDataset, DMRIRMatrixDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
from models.autoencoders import swapping_autoencoder as sae
from models.ema import EMA
from training.loggers.logger import Logger
from training.pipelines.pipeline import SupervisedPipeline
from functools import partial
from torchvision.models.squeezenet import squeezenet1_1
from torchvision.models.convnext import convnext_small, convnext_tiny, LayerNorm2d, Conv2dNormActivation
from models.utils import count_parameters
from models.autoencoders.convolutional import ConvEncoder


def prepare_train_val_test_sae(dataset_class, config, **dataset_kwargs):
    _, h, w = config["input_size"]
    train_transforms = A.Compose(
        [
            A.Resize(h, w),
            A.Affine(
                translate_percent=(0.125, 0.25), 
                rotate=(-30, 30), 
                scale=(0.8, 1.2), 
                shear=(-30, 30), 
                p=0.5
            ),
            A.ElasticTransform(),
            A.GaussianBlur(),
            A.Normalize(config["mean"], config["std"]),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image", "mask0": "mask"},
    )

    test_transforms = A.Compose(
        [
            A.Resize(h, w), 
            A.Normalize(config["mean"], config["std"]), 
            ToTensorV2()
        ],
        additional_targets={"image0": "image", "mask0": "mask"},
    )

    normal_ds = dataset_class("./data/dmrir/train/normal", transforms=test_transforms, **dataset_kwargs)
    normal_ds_train, normal_ds_val = normal_ds.split(0.7)
    normal_ds_train.transforms = train_transforms

    ano_ds = dataset_class("./data/dmrir/train/anomalous", transforms=test_transforms, **dataset_kwargs)
    ano_ds_train, ano_ds_val = ano_ds.split(0.7)
    ano_ds_train.transforms = train_transforms

    normal_ds_test = dataset_class("./data/dmrir/test/normal", transforms=test_transforms, **dataset_kwargs)
    ano_ds_test = dataset_class("./data/dmrir/test/anomalous", transforms=test_transforms, **dataset_kwargs)

    train_set = ConcatDataset([SupervisedWrapper(normal_ds_train, 0), SupervisedWrapper(ano_ds_train, 1)])
    val_set = ConcatDataset([SupervisedWrapper(normal_ds_val, 0), SupervisedWrapper(ano_ds_val, 1)])
    test_set = ConcatDataset([SupervisedWrapper(normal_ds_test, 0), SupervisedWrapper(ano_ds_test, 1)])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    return train_loader, val_loader, test_loader
    

class SymSAEClassifier(nn.Module):
    def __init__(self, encoder, texture_dims, device='cuda'):
        """
        Linear probing class for the SAE model
        """
        super(SymSAEClassifier, self).__init__()
        self.encoder = encoder.eval()
        self.head = nn.Sequential(
            nn.Linear(texture_dims,1),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):
        l_x, r_x = x
        with torch.no_grad():
            _, l_t = self.encoder(l_x.to(self.device), run_str=False, multi_tex=False)
            _, r_t = self.encoder(r_x.to(self.device), run_str=False, multi_tex=False)
            feats = torch.abs(l_t - r_t)
        out = self.head(feats)
        return out

class SAEClassifier(nn.Module):
    def __init__(self, encoder, texture_dims, device='cuda'):
        super().__init__()
        self.encoder = encoder.eval()
        self.head = nn.Sequential(
            nn.Linear(texture_dims,1),
            nn.Sigmoid()
        )
        self.device = device
    
    def forward(self, x):
        with torch.no_grad():
            _, t = self.encoder(x.to(self.device), run_str=False, multi_tex=False)
        out = self.head(t)
        return out
    
# class ConvClassifier(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.classifier = nn.Sequential(
#             ConvEncoder(128, 32, 1),
#             nn.Flatten(),
#             nn.Linear(10368, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         x = x.cuda()
#         return self.classifier(x)          

class SupervisedWrapper(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __getitem__(self, index):
        return self.dataset[index], self.label

    def __len__(self):
        return len(self.dataset)
    



CHANNELS = 16
STRUCTURE_CHANNELS = 4
TEXTURE_CHANNELS = 1024
pipeline = SupervisedPipeline()
pipeline.init_pipeline("./configs/supervised_dmr.yaml")
for i in range(10):
    pipeline.config['run_name'] = f'sae_nosym_dmr_{i}'
    encoder = EMA(sae.encoders.PyramidEncoder(CHANNELS, STRUCTURE_CHANNELS, TEXTURE_CHANNELS, gray=True)).cuda()

    #checkpoint = torch.load('logs/sae_dmrir/checkpoints/sae_c16_s4_t1024_tunedloss_400000_final.pth')
    checkpoint = torch.load('./logs/sae_dmrir/checkpoints/sae_c16_s4_t1024_fullimg_400000_final.pth')

    encoder.load_state_dict(checkpoint["enc_ema"])
    model = SymSAEClassifier(encoder, TEXTURE_CHANNELS).eval().cuda()
    #model = SAEClassifier(encoder, TEXTURE_CHANNELS).eval().cuda()
    
    #model = ConvClassifier().cuda()
    print(count_parameters(model))
    #dataset = partial(DMRIRMatrixDataset, side='both', apply_mask=True, return_mask=False)
    dataset = partial(DMRIRLeftRightDataset, apply_mask=True, return_mask=False)

    train_loader, val_loader, test_loader = prepare_train_val_test_sae(dataset, pipeline.config)
    with Logger(pipeline.config) as logger:
        trainer = pipeline.prepare_trainer(model, logger)
        pipeline.run(trainer, train_loader, val_loader, test_loader)
