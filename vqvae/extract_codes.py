import torch
from PIL import Image
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from vqvae import VQVAE

device = 'cuda'


@torch.no_grad()
def extract_latents(checkpoint, data_dir):
    dataset = datasets.DatasetFolder(extensions=('png', 'jpg'), loader=Image.open, root=data_dir,
                                     transform=Compose([Resize((128, 128)), ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    state = torch.load(checkpoint, map_location=device)
    hyper_parameters = state['hyper_parameters']
    vqvae = VQVAE(**hyper_parameters).to(device)

    vqvae.load_state_dict(state['model'])
    vqvae.eval()
    for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = images.to(device)
        _, indices = vqvae.encode(images)
        torch.save(indices, f'../data/celeba_latents/{i}.pt')





checkpoint = './runs/celeba/checkpoints/vqvae_celeba_step_80900.pt'
data_dir = '/home/lleonard/Documents/datasets/img_align_celeba/'
extract_latents(checkpoint, data_dir)