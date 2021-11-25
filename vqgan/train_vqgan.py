import click
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from tqdm import tqdm

from vqgan.models.vqgan import make_model_from_config
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt


device = 'cuda'
def train(name, model, dataloader, optimizer, base_step, num_training_updates):
    loss_fn = torch.nn.MSELoss()
    pbar = tqdm(range(num_training_updates))
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
        loss = commit_loss + loss_fn(reconstructed, image)
        tb_writer.add_scalar('loss', loss, step)
        if step % 10 == 0:
            with torch.no_grad():
                latents, latents_indices, _ = model.encode(image[0].unsqueeze(0))
                im_plt = plt.imshow(latents_indices.squeeze(0).cpu().numpy())
                image_indices = TF.ToTensor()(Image.fromarray(np.uint8(im_plt.get_cmap()(im_plt.get_array()) * 255)).convert('RGB').resize((128, 128))).to(device)
                tb_writer.add_image('reconstruction', torchvision.utils.make_grid([image[0], reconstructed[0], image_indices]), step)
        if step % 100 == 0:
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))
        pbar.set_description(str(round(loss.detach().item(), 4)))
        loss.backward()
        optimizer.step()

@click.command()
@click.option('--name', default='celeba', help='Name of the run')
@click.option('--config', default='./config/celeba_vqvae.yml', help='Path to the config file')
@click.option('--data-dir', type=click.Path(exists=True), default='/home/lleonard/Documents/datasets/img_align_celeba/', help='Path to the dataset')
@click.option('--batch-size', type=int, default=32)
def main(name, data_dir, batch_size, config):
    model = make_model_from_config(config).to(device)
    dataset = datasets.DatasetFolder(extensions=('png', 'jpg'), loader=Image.open, root=data_dir,
                                     transform=TF.Compose([TF.Resize((128, 128)), TF.ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train(name, model, dataloader, torch.optim.Adam(model.parameters(), lr=1e-4), 0, 1000)


if __name__ == '__main__':
    main()