import random

import click
import torch
from omegaconf import OmegaConf
from torch.nn.functional import embedding

import vqgan.models.vqgan
from gpt.gpt import GPTConfig, GPT
from gpt.utils import sample
import sys
import vqvae
import torchvision.transforms as TF

from vqgan.train_gpt import make_gpt

device = 'cuda'

@click.command()
@click.option("--checkpoint-path")
@click.option("--config-path")
def generate(checkpoint_path, config_path):
    config = OmegaConf.load(config_path)

    gpt_config = GPTConfig(**config.model.gpt.params)
    model, _, _ = make_gpt(checkpoint_path, gpt_config)
    model = model.to(device)
    model.eval()

    vqgan_model = vqgan.models.vqgan.make_model_from_config(config.model.vqgan).to(device)
    vqgan_model.load_from_file(config.model.vqgan.checkpoint_path)
    vqgan_model.eval()

    generated = sample(model,
                       torch.tensor([random.randint(0, 512)]).unsqueeze(0).to(device),
                       1023, temperature=1.0, sample=True,).view(-1, 32, 32)
    print(generated)
    embeds = vqgan_model.vq.embedding(generated).permute(0, 3, 1, 2)
    image = vqgan_model.decode(embeds).squeeze(0)
    image = torch.clamp(image, -1, 1)
    image = (image + 1) / 2
    TF.ToPILImage()(image).show()


if __name__ == "__main__":
    generate()
