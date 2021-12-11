import random

import click
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from omegaconf import OmegaConf

import vqgan.models.vqgan
from gpt.gpt import GPTConfig
from gpt.utils import sample

import torchvision.transforms as TF

from vqgan.train_gpt import make_gpt

device = 'cuda'

@click.command()
@click.option("--checkpoint-path")
@click.option("--config-path")
@click.option("--source-image", default=None)
def generate(checkpoint_path, config_path, source_image):
    config = OmegaConf.load(config_path)

    gpt_config = GPTConfig(**config.model.gpt.params)
    model, _, _ = make_gpt(checkpoint_path, gpt_config)
    model = model.to(device)
    model.eval()

    vqgan_model = vqgan.models.vqgan.make_model_from_config(config.model.vqgan).to(device)
    vqgan_model.load_from_file(config.model.vqgan.checkpoint_path)
    vqgan_model.eval()


    image = Image.open(source_image).convert('RGB').resize((256, 256))
    # image_chunk = chunk_image(image, 256)
    # reconstructed_chunks = recursive_map(functools.partial(reconstruct_image, model=model), tqdm(image_chunk))
    # reconstructed_image = reassemble_image(reconstructed_chunks, 256)
    # reconstructed_image.show()
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).to(device)

    quantized, latent_indices, _ = vqgan_model.encode(image.unsqueeze(0))
    TF.ToPILImage()((vqgan_model.decode(quantized).squeeze(0) + 1) / 2).show()
    latent_indices = torch.flatten(latent_indices.squeeze(0))
    seed = latent_indices[0:100].unsqueeze(0).repeat_interleave(8, dim=0)
    #seed = torch.clamp((torch.randn((8, 1)).to(device).to(device) * 8192).long(), 0, 8190)
    print(seed.shape)
    generated = sample(model,
                      seed,
                       256 - seed.shape[-1], temperature=1.0, sample=True,).view(-1, 16, 16)
    print(generated.shape)
    embeds = vqgan_model.vq.embedding(generated).permute(0, 3, 1, 2)
    image = vqgan_model.decode(embeds).squeeze(0)
    image = torch.clamp(image, -1, 1)
    image = (image + 1) / 2
    TF.ToPILImage()(torchvision.utils.make_grid(image)).show()


if __name__ == "__main__":
    generate()
