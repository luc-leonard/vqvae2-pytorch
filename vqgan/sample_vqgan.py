import datetime
import random
from typing import List, TypeVar, Iterable, Callable

import PIL.Image
import click
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from omegaconf import OmegaConf
from torch.nn.functional import embedding

import torchvision.transforms as TF
from tqdm import tqdm

from vqgan.models.vqgan import make_model_from_config
import functools

def chunk_image(image: PIL.Image.Image, chunk_size: int) -> List[List[Image.Image]]:
    """
    Splits an image into smaller images.

    :param image: The image to split.
    :param chunk_size: The size of the smaller images.
    :return: The list of smaller images.
    """
    chunks = []
    for i in range(0, image.height, chunk_size):
        row = []
        for j in range(0, image.width, chunk_size):
            row.append(image.crop((j, i, j + chunk_size, i + chunk_size)))
        chunks.append(row)
    return chunks


def reassemble_image(chunks: List[List[Image.Image]], chunk_size: int) -> Image.Image:
    """
    Reassembles a list of smaller images into a bigger image.

    :param chunks: The list of smaller images.
    :param chunk_size: The size of the smaller images.
    :return: The reassembled image.
    """
    image = Image.new('RGB', (chunk_size * len(chunks[0]), chunk_size * len(chunks)))
    for i, row in enumerate(chunks):
        for j, chunk in enumerate(row):
            image.paste(chunk, (j * chunk_size, i * chunk_size))
    return image


def reconstruct_image(image, model) -> Image:
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1)
    quantized, latent_indices, _ = model.encode(image.unsqueeze(0))
    print(latent_indices.shape)
    reconstructed = model.decode(quantized).squeeze(0)
    reconstructed = torch.clamp(reconstructed, -1, 1)
    reconstructed = (reconstructed + 1) / 2
    return TF.ToPILImage()(reconstructed)


T = TypeVar('T')

def recursive_map(f: Callable[[T], T], i: Iterable[Iterable[T]]) -> List[List[T]]:
    return [recursive_map(f, x) if isinstance(x, Iterable) else f(x) for x in i]


@click.command()
@click.option("--checkpoint-path", "-v")
@click.option("--config", "-c")
@click.option("--image-path", "-c")
@click.option("--device", "-d", default="cpu")
@torch.no_grad()
def generate(checkpoint_path, config, image_path, device='cpu'):
    config = OmegaConf.load(config)
    model = make_model_from_config(config.model).to(device)
    model.load_from_file(checkpoint_path)
    model.eval()


    image = Image.open(image_path).convert('RGB')
    # image_chunk = chunk_image(image, 256)
    # reconstructed_chunks = recursive_map(functools.partial(reconstruct_image, model=model), tqdm(image_chunk))
    # reconstructed_image = reassemble_image(reconstructed_chunks, 256)
    # reconstructed_image.show()
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).to(device)
    print('before encode', datetime.datetime.now())
    quantized, latent_indices, _ = model.encode(image.unsqueeze(0))
    print(latent_indices.shape)
    print('before decode', datetime.datetime.now())
    reconstructed = model.decode(quantized).squeeze(0)
    print('displaying', datetime.datetime.now())
    reconstructed = torch.clamp(reconstructed, -1, 1)
    reconstructed = (reconstructed + 1) / 2
    TF.ToPILImage()(torchvision.utils.make_grid([reconstructed])).show()
    Image.open(image_path).convert('RGB').show()


if __name__ == "__main__":
    generate()
