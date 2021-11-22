import io

import numpy as np
import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from PIL import Image

from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage
from tqdm import tqdm
import matplotlib.pyplot as plt
import click

from vqvae import VQVAE

device = 'cuda'


def train(model, dataloader, optimizer, base_step, num_training_updates, hyper_parameters):
    loss_fn = torch.nn.MSELoss()
    pbar = tqdm(range(num_training_updates), ncols=100)
    tb_writer = SummaryWriter('./runs/celeba/logs')

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
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step, 'hyper_parameters': hyper_parameters}, "vqvae_celeba_step_{}.pt".format(step))
        pbar.set_description(str(round(loss.detach().item(), 4)))
        loss.backward()
        optimizer.step()


# these will be used if we want to train the model from scratch
batch_size = 128
num_training_updates = 15000
input_dim = 3
hidden_dim = 256
dim_residual_layer = 32
num_residual_layer = 8
embedding_dim = 128
codebook_size = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3


@click.command()
@click.option('--resume_from', default=None, help='Resume training from the model in the given path')
@click.option('--data_dir', default='/home/lleonard/Documents/datasets/img_align_celeba/', help='Path to the dataset')
def main(resume_from, data_dir='data/celeba'):
    if not resume_from:
        hyper_parameters = {
            'batch_size': batch_size,
            'num_training_updates': num_training_updates,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'dim_residual_layer': dim_residual_layer,
            'num_residual_layer': num_residual_layer,
            'embedding_dim': embedding_dim,
            'codebook_size': codebook_size,
            'commitment_cost': commitment_cost,
            'decay': decay,
            'learning_rate': learning_rate}
        current_step = 0
        vqvae = VQVAE(**hyper_parameters).to(device)
        opt = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    else:
        state = torch.load(resume_from)
        hyper_parameters = state['hyper_parameters']
        vqvae = VQVAE(**hyper_parameters).to(device)
        opt = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)

        vqvae.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        current_step = state['step']


    dataset = datasets.DatasetFolder(extensions=('png', 'jpg'), loader=Image.open, root=data_dir, transform=Compose([Resize((128, 128)), ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    vqvae.train()

    train(vqvae, dataloader, opt, current_step, num_training_updates, hyper_parameters)


if __name__ == "__main__":
    print('.')
    main()