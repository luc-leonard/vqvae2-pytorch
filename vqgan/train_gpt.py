import glob
import random
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf
from torch.nn.functional import embedding
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import vqgan.models.vqgan
from gpt.gpt import GPT, GPTConfig
from gpt.trainer import Trainer, TrainerConfig
from gpt.utils import sample
import torchvision.transforms as TF

from utils.data import LatentsDataset
from utils.utils import get_class_from_str

device = 'cuda'

def make_callback(config, name, vqgan_model):
    tb_writer = SummaryWriter(f'./runs/{name}/logs')
    Path(f'./runs/{name}').mkdir(parents=True, exist_ok=True)

    def callback(step, model,  loss, optimizer, x, y):
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

            generated_latents_indices = sample(raw_model, torch.tensor([y[0, 0]]).unsqueeze(0).to(device), 1023, sample=True)\
                .view(1, 32, 32)\
                .to('cpu')

            generated_latents = vqgan_model.vq.embedding(generated_latents_indices).permute(0, 3, 1, 2)
            upscaled_latents_indices = torch.nn.Upsample(size=(256, 256))(generated_latents_indices.unsqueeze(1).float()).squeeze(1).long()

            upscaled_latents_indices = torch.stack([upscaled_latents_indices, upscaled_latents_indices, upscaled_latents_indices]).permute(1, 0, 2, 3)
            image = vqgan_model.decode(generated_latents)
            image = torch.clamp(image, -1, 1)
            image = (image + 1) / 2
            tb_writer.add_image('sample', torch.cat([image.squeeze(0), upscaled_latents_indices.squeeze(0)], 1), step)

            # y lacks the first token :)
            y = torch.cat([y.cpu(), torch.tensor([16]).repeat(8).view(-1, 1)], dim=-1).view(-1, 32, 32)[:1, :, :]
            upscaled_y = torch.nn.Upsample(size=(256, 256))(y.float().unsqueeze(0)).squeeze(0).squeeze(0).long()
            embeds = vqgan_model.vq.embedding(y).permute(0, 3, 1, 2)
            real_image = vqgan_model.decode(embeds).squeeze(0)
            real_image = torch.clamp(real_image, -1, 1)
            real_image = (real_image + 1) / 2

            upscaled_y = torch.stack([upscaled_y, upscaled_y, upscaled_y])
            print(real_image.shape, upscaled_y.shape)
            tb_writer.add_image('real', torch.cat([real_image, upscaled_y], dim=1), step)
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
@click.option('--resume-from', default=None)
@click.option('--config-path', default='')
@torch.no_grad()
def train_gpt(name, resume_from, config_path):
    config = OmegaConf.load(config_path)
    vqgan_model = vqgan.models.vqgan.make_model_from_config(config.model.vqgan).eval()
    vqgan_model.load_from_file(config.model.vqgan.checkpoint_path)
    dataset = get_class_from_str(config.data.target)(**config.data.params)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    print(len(train_dataset))
    print(len(test_dataset))
    gpt_config = GPTConfig(**config.model.gpt.params)
    model, opt_data, base_step = make_gpt(resume_from, gpt_config)
    print(base_step)



    trainer = Trainer(model, train_dataset, test_dataset, config=TrainerConfig(
        ckpt_path=f'./runs/{name}/checkpoint.pt',
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        max_epochs=10),
        callback=make_callback(config.training, name, vqgan_model))
    trainer.train(base_step, opt_data)
    trainer.save_checkpoint()




if __name__ == '__main__':
    train_gpt()
