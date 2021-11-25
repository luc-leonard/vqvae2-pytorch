import random

import click
import torch
from torch.nn.functional import embedding

import torchvision.transforms as TF

from vqvae import load_vqvae


@click.command()
@click.option("--vqvae-path", "-v")
@click.option("--latent-path", "-l")
@click.option("--index", "-i", default=0)
def generate(vqvae_path, latent_path, index, device='cpu'):
    model = load_vqvae(vqvae_path).to(device).eval()
    latents = torch.load(latent_path, map_location=device)

    embeds = embedding(latents, model.vq._codebook.embed)[index]
    TF.ToPILImage()(model.decode(embeds.unsqueeze(0)).squeeze(0)).show()


if __name__ == "__main__":
    generate()
