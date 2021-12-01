from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from multiprocessing import Queue, Process
from utils.utils import get_class_from_str
from vqgan.models.vqgan import make_model_from_config

device = 'cuda'


@click.command()
@click.option('--config', '-c', type=str, required=True)
@click.option('--checkpoint-path', type=str, required=True)
@click.option('--out-path', type=str, required=True)
@click.option('--batch-size', type=int, required=False, default=64)
@torch.no_grad()
def extract_latents(config, checkpoint_path, out_path, batch_size=64):
    config = OmegaConf.load(config)
    model = make_model_from_config(config.model).to(device)
    model.load_from_file(checkpoint_path)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    dataset = get_class_from_str(config.data.target)(**config.data.params)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = images.to(device)
            _, indices, _ = model.encode(images)
            torch.save(indices, f'{out_path}/{i}.pt')


if __name__ == '__main__':
    extract_latents()