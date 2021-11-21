import io

import numpy as np
import torch
import torchvision.utils
from fastai.data.load import DataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from PIL import Image

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
from tqdm import tqdm
import matplotlib.pyplot as plt
import click

from vqvae import VQVAE

device = 'cuda'


def train(model, dataloader, optimizer, base_step, num_training_updates):
    loss_fn = torch.nn.MSELoss()
    pbar = tqdm(range(num_training_updates), ncols=100)
    tb_writer = SummaryWriter('./logs2')

    data_iterator = iter(dataloader)
    for step in pbar:
        step = base_step + step
        try:
            (image, _) = next(data_iterator)
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
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step}, "vqvae_celeba_step_{}.pt".format(step))
        pbar.set_description(str(round(loss.detach().item(), 4)))
        loss.backward()
        optimizer.step()


batch_size = 128
num_training_updates = 15000
input_dim = 3
num_hiddens = 256
num_residual_hiddens = 32
num_residual_layers = 8
embedding_dim = 128
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3


@click.command()
@click.option('--resume_from', default=None, help='Resume training from the model in the given path')
@click.option('--data_dir', default='/home/lleonard/Documents/datasets/img_align_celeba/', help='Resume training from the model in the given path')
def main(resume_from, data_dir='data/celeba'):
    vqvae = VQVAE(input_dim=input_dim, hidden_dim=num_hiddens, embedding_dim=embedding_dim, codebook_size=num_embeddings, num_residual_layer=num_residual_layers, dim_residual_layer=num_residual_hiddens).to(device)
    opt = torch.optim.Adam(vqvae.parameters(), lr=1e-3)

    state = torch.load(resume_from) if resume_from else None
    current_step = 0
    if state:
        vqvae.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        current_step = state['step']

    dataset = datasets.DatasetFolder(extensions=('png', 'jpg'), loader=Image.open, root=data_dir, transform=Compose([Resize((128, 128)), ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    vqvae.train()

    train(vqvae, dataloader, opt, current_step, num_training_updates)

if __name__ == "__main__":
    main()