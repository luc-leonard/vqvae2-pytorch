import click
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from tqdm import tqdm

import utils.data
from utils.utils import get_class_from_str
from vqgan.models.vqgan import make_model_from_config
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt
import lpips

from vqgan.modules.losses import CombinedLosses

device = 'cuda'


def make_loss(config):
    return


def make_callbacks(config, run_dir):
    callbacks = []
    for callback_config in config.train.callbacks:
        callbacks.append(get_class_from_str(callback_config.target)(run_dir=run_dir,
                                                                    **(callback_config.params or {})))
    return callbacks

def train(name, model, dataloader, optimizer, base_step, config):
    pbar = tqdm(range(config.train.total_steps))

    run_dir = f'./runs/{name}'
    # tb_writer = SummaryWriter(f'./runs/{name}/logs')

    loss_fn = CombinedLosses(config.loss, optimizer)
    callbacks = make_callbacks(config, run_dir)
    data_iterator = iter(dataloader)
    for step in pbar:
        step = base_step + step
        try:
            (image, _) = next(data_iterator)
        except Exception as e:
            data_iterator = iter(dataloader)
            continue
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        optimizer.zero_grad()
        image = image.to(device)
        model_output = model(image)
        loss, log, current_opt = loss_fn(model_output, image, step, model.get_last_layer().weight)
        loss.backward()
        current_opt.step()

        for callback in callbacks:
            callback.on_step(model, image, image, model_output, log, step)

        if step % config.train.save_every == 0:
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))

        pbar.set_description(str(round(loss.detach().item(), 4)))


    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))
    return step


@click.command()
@click.option('--name', default='celeba', help='Name of the run')
@click.option('--config', default='./config/celeba_vqvae.yml', help='Path to the config file')
@click.option('--resume-from', default=None, help='checkpoint')
@click.option('--epochs', default=1, help='base step')
def main(name, config, resume_from, epochs):
    config = OmegaConf.load(config)
    model = make_model_from_config(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    base_step = 0

    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        base_step = checkpoint['step']

    dataset = get_class_from_str(config.data.target)(**config.data.params)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=1)
    for _ in range(epochs):
        print('.')
        base_step = train(name, model, dataloader, optimizer, base_step, config)


if __name__ == '__main__':
    main()