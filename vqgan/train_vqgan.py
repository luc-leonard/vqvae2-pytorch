import click
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from tqdm import tqdm

from vqgan.models.vqgan import make_model_from_config
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt
import lpips

device = 'cuda'
def train(name, model, dataloader, optimizer, base_step, config):
    loss_fn = torch.nn.MSELoss()

    if config.loss.perceptual_loss > 0:
        loss_lpips = lpips.LPIPS().to(device)
    else:
        loss_lpips = None

    pbar = tqdm(range(config.train.total_steps))
    tb_writer = SummaryWriter(f'./runs/{name}/logs')

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
        loss = (commit_loss * config.loss.codebook_loss) + \
               (loss_fn(reconstructed, image) * config.loss.pixel_loss)
        if loss_lpips:
            loss += (loss_lpips(reconstructed, image, normalize=True) * config.loss.perceptual_loss).mean()
        loss.backward()
        optimizer.step()

        tb_writer.add_scalar('loss', loss, step)
        if step % config.train.sample_every == 0:
            with torch.no_grad():
                latents, latents_indices, _ = model.encode(image)
                latents_indices = torch.nn.Upsample(size=(128,128))(latents_indices.unsqueeze(1).float()).squeeze(1).long()

                latents_indices = torch.stack([latents_indices, latents_indices, latents_indices]).permute(1, 0, 2, 3)

                tb_writer.add_image('reconstruction',
                                    torchvision.utils.make_grid(torch.cat([image, reconstructed, latents_indices]), nrow=13),
                                    step)
        if step % config.train.save_every == 0:
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))
        pbar.set_description(str(round(loss.detach().item(), 4)))
    return step


@click.command()
@click.option('--name', default='celeba', help='Name of the run')
@click.option('--config', default='./config/celeba_vqvae.yml', help='Path to the config file')
@click.option('--resume-from', default=None, help='checkpoint')
@click.option('--epochs', default=1, help='base step')
def main(name, config, resume_from, epochs):
    config = OmegaConf.load(config)
    model = make_model_from_config(config).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    base_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        base_step = checkpoint['step']
    dataset = datasets.DatasetFolder(extensions=('png', 'jpg'), loader=Image.open, root=config.data.dir,
                                     transform=TF.Compose([TF.Resize((128, 128)), TF.ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=4)
    for _ in range(epochs):
        base_step = train(name, model, dataloader, optimizer, base_step, config)


if __name__ == '__main__':
    main()