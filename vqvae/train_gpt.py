import glob
import random
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf
from torch.nn.functional import embedding
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from gpt.gpt import GPT, GPTConfig
from gpt.trainer import Trainer, TrainerConfig
from gpt.utils import sample
import torchvision.transforms as TF

from vqvae import load_vqvae

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


def make_callback(config, name, the_vqvae):
    tb_writer = SummaryWriter(f'./runs/{name}/logs')
    Path(f'./runs /{name}').mkdir(parents=True, exist_ok=True)

    def callback(step, model,  loss, optimizer):
        tb_writer.add_scalar('loss', loss, step)
        raw_model = model.module if hasattr(model, "module") else model

        if step % config.save_every == 0:
            print('saving model')
            torch.save({
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                        'opt': optimizer.state_dict(),
                        'step': step,
                        },
                f'./runs/{name}/gpt_model_checkpoint_{step}.pt')

        if step % config.sample_every == 0:
            print('generating sample')
            generated_latents = sample(raw_model, torch.tensor([random.randint(0, 512)]).unsqueeze(0).to(device), 1023).view(-1, 32, 32).to('cpu')
            image = the_vqvae.decode(embedding(generated_latents, the_vqvae.vq._codebook.embed)).squeeze(0)
            tb_writer.add_image('sample', image, step)
    return callback


def make_gpt(checkpoint_path=None, config=None):
    if checkpoint_path is None:
        model = GPT(config)
        opt_data = None
        base_step = 0
    else:
        data = torch.load(checkpoint_path, map_location='cpu')
        model = GPT(config)
        model.load_state_dict(data['model'])
        opt_data = data['opt']
        base_step = data['step']
    return model, opt_data, base_step


@click.command()
@click.option('--name', default=None)
@click.option('--vqvae-path', default=None)
@click.option('--resume-from', default=None)
@click.option('--config-path', default='')
@torch.no_grad()
def train_gpt(name, resume_from, config_path, vqvae_path):
    config = OmegaConf.load(config_path)
    dataset = LatentsDataset(root=config.data.dir)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    print(len(train_dataset))
    print(len(test_dataset))
    gpt_config = GPTConfig(**config.model.gpt.params)
    model, opt_data, base_step = make_gpt(resume_from, gpt_config)
    print(base_step)

    the_vqvae = load_vqvae(config.model.vqvae.params, vqvae_path)

    trainer = Trainer(model, train_dataset, test_dataset, config=TrainerConfig(
        ckpt_path=f'./runs/{name}/checkpoint.pt',
        batch_size=16,
        learning_rate=config.training.learning_rate,
        max_epochs=1),
        callback=make_callback(config.training, name, the_vqvae))
    trainer.train(base_step, opt_data)
    trainer.save_checkpoint()




if __name__ == '__main__':
    train_gpt()
