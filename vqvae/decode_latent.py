import random

import click
import torch
from omegaconf import OmegaConf
from torch.nn.functional import embedding

import torchvision.transforms as TF

from vqvae import load_vqvae


@click.command()
@click.option("--vqvae-path", "-v")
@click.option("--vqvae-config", "-c")
@click.option("--latent-path", "-l")
@click.option("--index", "-i", default=0)
def generate(vqvae_path, vqvae_config, latent_path, index, device='cpu'):
    config = OmegaConf.load(vqvae_config)
    model = load_vqvae(config.model.params, vqvae_path).to(device).eval()
    latents = torch.load(latent_path, map_location=device)

    embeds = embedding(latents, model.vq._codebook.embed)[index]
    TF.ToPILImage()(model.decode(embeds.unsqueeze(0)).squeeze(0)).show()


if __name__ == "__main__":
    generate()
