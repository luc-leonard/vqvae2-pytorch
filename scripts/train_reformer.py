import os
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf
from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import vqgan.models.vqgan
from gpt.gpt import GPT, GPTConfig
from gpt.trainer import Trainer, TrainerConfig
from gpt.utils import sample
from utils.utils import get_class_from_str
import shutil



device = 'cuda'

def make_callback(config, name, vqgan_model):
    tb_writer = SummaryWriter(f'./runs/{name}/logs')
    Path(f'./runs/{name}').mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def callback(step, model,  loss, optimizer, x, y):
        tb_writer.add_scalar('loss', loss, step)
        raw_model = model.module if hasattr(model, "module") else model

        if step % config.training.save_every == 0:
            print('saving model')
            torch.save({
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                        'opt': optimizer.state_dict(),
                        'step': step,
                        },
                f'./runs/{name}/gpt_model_checkpoint_{step}_.pt')
            shutil.move(f'./runs/{name}/gpt_model_checkpoint_{step}_.pt', f'./runs/{name}/gpt_model_checkpoint_{step}.pt')
            file_name = f'./runs/{name}/gpt_model_checkpoint_latest.pt'

            os.symlink(f'gpt_model_checkpoint_{step}.pt', file_name + '.tmp')
            os.rename(file_name + '.tmp', file_name)
        if step % (config.training.sample_every * 10) == 0:
            print('sampling')
            tb_writer.add_image('full image', (model, vqgan_model), step)
        if step % config.training.sample_every == 0:
            print('generating sample')
            if x.shape != y.shape:
                # if input is larger than output, it means that it is prepended with conditioning
                seed = x[0, :config.data.params.patch_size[0] * config.data.params.patch_size[1] + 1].unsqueeze(0) # seed must contains the coordinates + 1st token
            else:
                seed = y[0, 0:1].unsqueeze(0) #seed just contains the first latent indice

            generated_latents_indices = sample(raw_model, seed.to(device), 255, sample=True)
            if generated_latents_indices.shape[1] > 256:
                generated_latents_indices = generated_latents_indices[:, 256:]
            generated_latents_indices = generated_latents_indices.squeeze(1)
            generated_latents_indices = generated_latents_indices\
                .view(1, 16, 16)

            generated_latents_indices = torch.clamp(generated_latents_indices, 0, 8191) # untrained GPT can produce tokens which are part of the `coord` vocab
            generated_latents = vqgan_model.vq.embedding(generated_latents_indices).permute(0, 3, 1, 2)
            upscaled_latents_indices = torch.nn.Upsample(size=(256,256))(generated_latents_indices.unsqueeze(1).float()).squeeze(1).long()
            upscaled_latents_indices = torch.stack([upscaled_latents_indices, upscaled_latents_indices, upscaled_latents_indices]).permute(1, 0, 2, 3)

            image = vqgan_model.decode(generated_latents)
            image = torch.clamp(image, -1, 1)
            image = (image + 1) / 2
            tb_writer.add_image('sample', torch.cat([image.squeeze(0), upscaled_latents_indices.squeeze(0)], 1), step)

            # y lacks the first token :)
            y = torch.cat([y, torch.tensor([16]).repeat(config.training.batch_size).view(-1, 1).to(device)], dim=-1).view(-1, 16, 16)[:1, :, :]
            upscaled_y = torch.nn.Upsample(size=(256, 256))(y.float().unsqueeze(0)).squeeze(0).squeeze(0).long()
            embeds = vqgan_model.vq.embedding(y).permute(0, 3, 1, 2)
            real_image = vqgan_model.decode(embeds).squeeze(0)
            real_image = torch.clamp(real_image, -1, 1)
            real_image = (real_image + 1) / 2

            upscaled_y = torch.stack([upscaled_y, upscaled_y, upscaled_y])
            #print(real_image.shape, upscaled_y.shape)
            tb_writer.add_image('real', torch.cat([real_image, upscaled_y], dim=1), step)
    return callback


@click.command()
@click.option('--name', default=None)
@click.option('--resume-from', default=None)
@click.option('--config-path', default='')
def train_reformer(name, resume_from, config_path):
    import shutil
    Path(f'./runs/{name}').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, f'./runs/{name}/config.yaml')
    config = OmegaConf.load(config_path)
    vqgan_model = vqgan.models.vqgan.make_model_from_config(config.model.vqgan).eval().to(device)
    vqgan_model.load_from_file(config.model.vqgan.checkpoint_path)
    dataset = get_class_from_str(config.data.target)(**config.data.params)

   # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    reformer = ReformerLM(
        num_tokens=8192,
        dim=1024,
        depth=12,
        max_seq_len=4096,
        lsh_dropout=0.1,
        causal=True,
        full_attn_thres=1024
    ).to(device)

    optimizer = torch.optim.Adam(reformer.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=6)
    model = TrainingWrapper(reformer, ignore_index=0, pad_value=0)
    pbar = tqdm(dataloader)
    step = 0
    for (x, _) in pbar:
        x = x.to(device)
        x = torch.flatten(x, start_dim=1)
        optimizer.zero_grad()
        loss = model(x, return_loss=True)
        pbar.set_description(f'Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()

        if step % config.training.save_every == 0:
            seq = model.generate(x[0][0].unsqueeze(0), 4095)
            print(seq.shape)
            torch.save({
                'model': reformer.state_dict(),
                'optimizer': optimizer.state_dict(),}
                , f'./runs/{name}/reformer_{step}.pt')
        step += 1


if __name__ == '__main__':
    train_reformer()
