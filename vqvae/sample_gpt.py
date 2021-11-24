import random

import click
import torch
from torch.nn.functional import embedding

from gpt.gpt import GPTConfig, GPT
from gpt.utils import sample
import sys
import vqvae
import torchvision.transforms as TF

@click.command()
@click.option("--checkpoint-path", "-c")
@click.option("--vqvae-path", "-v")
def generate(checkpoint_path, vqvae_path, device='cpu'):
    config = GPTConfig(vocab_size=512, block_size=32*32, n_head=8, n_layer=12, n_embd=512, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0)
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    vqvae_state = torch.load(vqvae_path, map_location=device)
    vqvae_model = vqvae.VQVAE(**vqvae_state['hyper_parameters']).to(device)
    vqvae_model.load_state_dict(vqvae_state['model'])

    generated = sample(model, torch.tensor([random.randint(0, 512)]).unsqueeze(0).to(device), 1023).view(-1, 32, 32)
    print(generated)
    embeds = embedding(generated, vqvae_model.vq._codebook.embed)
    TF.ToPILImage()(vqvae_model.decode(embeds).squeeze(0)).show()


if __name__ == "__main__":
    generate()
