import models.autoencoders.swapping_autoencoder as sae
from training.pipelines.dmrir_pipeline import SAEDMRIRPipeline


pipeline = SAEDMRIRPipeline()
pipeline.init_pipeline("./configs/sae_dmr_ir.yaml")

config = pipeline.get_config()

CHANNELS = config["channels"]
STRUCTURE_CHANNELS = config["struct_channels"]
TEXTURE_CHANNELS = config["text_channels"]

encoder = sae.encoders.PyramidEncoder(
    CHANNELS,
    structure_channel=STRUCTURE_CHANNELS,
    texture_channel=TEXTURE_CHANNELS,
    gray=True,
).to(config["device"])

generator = sae.generators.Generator(
    CHANNELS,
    structure_channel=STRUCTURE_CHANNELS,
    texture_channel=TEXTURE_CHANNELS,
    gray=True,
).to(config["device"])

str_projectors = sae.layers.MultiProjectors(
    [CHANNELS, CHANNELS * 2, CHANNELS * 8], use_mlp=True
).to(config["device"])

discriminator = sae.discriminators.Discriminator(
    config["input_size"][-1], channel_multiplier=1, gray=True
).to(config["device"])

cooccur = sae.discriminators.CooccurDiscriminator(
    CHANNELS, size=config["input_size"][-1] * config["max_patch_size"], gray=True
).to(config["device"])

normal_loader, val_loader = pipeline.prepare_data()

trainer = pipeline.prepare_trainer(
    encoder, generator, str_projectors, discriminator, cooccur
)

pipeline.run(trainer, normal_loader, val_loader)
