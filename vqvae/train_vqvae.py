import io
from pathlib import Path

import numpy as np
import torch
import torchvision.utils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from PIL import Image

from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
from tqdm import tqdm
import matplotlib.pyplot as plt
import click

from utils.data import MyImageFolderDataset
from vqvae import VQVAE

device = 'cuda'


def train(name, model, dataloader, optimizer, base_step, num_training_updates, config):
    loss_fn = torch.nn.MSELoss()
    pbar = tqdm(range(num_training_updates), ncols=100)
    path = Path(f'./runs/{name}')
    path.mkdir(exist_ok=True, parents=True)

    tb_writer = SummaryWriter(f'./runs/{name}/logs')

    data_iterator = iter(dataloader)
    for step in pbar:
        step = base_step + step
        try:
            image = next(data_iterator)
        except Exception as e:
            data_iterator = iter(dataloader)
            continue
        optimizer.zero_grad()
        image = image.to(device)
        commit_loss, reconstructed = model(image)
        loss = commit_loss + loss_fn(reconstructed, image)
        tb_writer.add_scalar('loss', loss, step)
        if step % 10 == 0:
            with torch.no_grad():
                latents, latents_indices = model.encode(image[0].unsqueeze(0))
                im_plt = plt.imshow(latents_indices.squeeze(0).cpu().numpy())
                image_indices = ToTensor()(Image.fromarray(np.uint8(im_plt.get_cmap()(im_plt.get_array()) * 255)).convert('RGB').resize((128, 128))).to(device)
                tb_writer.add_image('reconstruction', torchvision.utils.make_grid([image[0], reconstructed[0], image_indices]), step)
        if step % 100 == 0:
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step, 'config': config}, str(path / "vqvae_celeba_step_{}.pt".format(step)))
        pbar.set_description(str(round(loss.detach().item(), 4)))
        loss.backward()
        optimizer.step()


@click.command()
@click.option('--name', default='celeba_2')
@click.option('--config-file', default='', help='config file')
@click.option('--resume-from', default=None, help='Resume training from the model in the given path')
def main(name, config_file, resume_from):
    config = OmegaConf.load(config_file)
    vqvae = VQVAE(**config.model.params).to(device)
    opt = torch.optim.Adam(vqvae.parameters(), lr=config.training.learning_rate)
    current_step = 0

    if resume_from:
        state = torch.load(resume_from, map_location='cpu')
        vqvae.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        current_step = state['step']


    dataset = MyImageFolderDataset(extensions=('png', 'jpg'), data_dir=config.data.dir, transform=Compose([Resize((128, 128)), ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=4)
    vqvae.train()

    train(name, vqvae, dataloader, opt, current_step, config.training.total_steps, config)


if __name__ == "__main__":
    print('.')
    main()