import datetime

import click
import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.utils
from PIL import Image
from omegaconf import OmegaConf

from vqgan.models.vqgan import make_model_from_config


@click.command()
@click.option("--checkpoint-path", "-v")
@click.option("--config", "-c")
@click.option("--image-path", "-c")
@click.option("--device", "-d", default="cuda")
@torch.no_grad()
def generate(checkpoint_path, config, image_path, device='cpu'):
    config = OmegaConf.load(config)
    model = make_model_from_config(config.model).to(device)
    model.load_from_file(checkpoint_path)
    model.eval()


    image = Image.open(image_path).convert('RGB')

    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).to(device)
    print('before encode', datetime.datetime.now())
    quantized, latent_indices, _ = model.encode(image.unsqueeze(0))
    print(quantized.shape)
    print('before decode', datetime.datetime.now())
    reconstructed = model.decode(quantized).squeeze(0)
    print('displaying', datetime.datetime.now())
    reconstructed = torch.clamp(reconstructed, -1, 1)
    reconstructed = (reconstructed + 1) / 2
    TF.ToPILImage()(torchvision.utils.make_grid([reconstructed])).show()
    Image.open(image_path).convert('RGB').show()


if __name__ == "__main__":
    generate()
