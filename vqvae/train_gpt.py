import glob
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from gpt.gpt import GPT, GPT1Config, GPTConfig
from gpt.trainer import Trainer, TrainerConfig
from vqvae import VQVAE
from torch.nn import functional as F

device = 'cuda'

class LatentsDataset(Dataset):
    def __init__(self, root):
        self.root_dir = root
        files = glob.glob(root + '/*.pt')
        self.size = 0
        self.files = []
        for file in files:
            self.files.append(torch.load(file, map_location='cpu'))
            self.size += self.files[-1].shape[0]

        # none is a 'last file' that would not contain the same amount as the others
        if self.files[0].shape[0] == self.files[-1].shape[0]:
            for idx, file in enumerate(self.files):
                if file.shape[0] != self.files[0].shape[0]:
                    # this is out last file. put it at the end of the list
                    self.files = self.files[:idx] + self.files[idx+1:] + [file]
        else:
            if self.files[0].shape[0] < self.files[-1].shape[0]:
                self.files = self.files[1:] + self.files[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item_per_file = self.files[0].shape[0]
        file_idx = idx // item_per_file
        item_idx = idx % item_per_file
        value = torch.flatten(self.files[file_idx][item_idx])
        return value[:- 1], value[1:]


@torch.no_grad()
def train_gpt(data_dir):
    dataset = LatentsDataset(root=data_dir)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    print(len(train_dataset))
    print(len(test_dataset))
    config = GPTConfig(vocab_size=512, block_size=32*32, n_head=8, n_layer=12, n_embd=512, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0)
    model = GPT(config).to(device)
    print(model)
    trainer = Trainer(model, test_dataset, test_dataset, config=TrainerConfig(
        ckpt_path='./runs/celeba_gpt/',
        batch_size=16,
        max_epochs=1))
    trainer.train()
    trainer.save_checkpoint()





# checkpoint = './runs/celeba/checkpoints/vqvae_celeba_step_80900.pt'
data_dir = '../data/celeba_latents/'
train_gpt(data_dir)