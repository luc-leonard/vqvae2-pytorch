from pathlib import Path

import click
import torch
import torchvision.utils
import tqdm
from omegaconf import OmegaConf
from perceiver_pytorch import PerceiverIO, PerceiverLM
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import vqgan.models.vqgan
from utils.utils import get_class_from_str

device = 'cuda'

@click.command()
@click.option('--config-path', default='')
@click.option('--name', default='')
def main(config_path, name):
    model = PerceiverLM(
        dim=1,  # dimension of sequence to be encoded
        #queries_dim=1,  # dimension of decoder queries
        #logits_dim=8192,  # dimension of final logits
        depth=6,  # depth of net
        num_latents=256,
        num_tokens=8192,
        max_seq_len=256,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=512,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=64,  # number of dimensions per cross attention head
        latent_dim_head=64,  # number of dimensions per latent self attention head
        weight_tie_layers=False  # whether to weight tie layers (optional, as indicated in the diagram)
    ).to(device)
    Path(f'./runs/{name}').mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)
    vqgan_model = vqgan.models.vqgan.make_model_from_config(config.model.vqgan).eval().to(device)
    vqgan_model.load_from_file(config.model.vqgan.checkpoint_path)
    dataset = get_class_from_str(config.data.target)(**config.data.params)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    pbar = tqdm.tqdm(dataloader)
    step = 0

    tb_writer = SummaryWriter(f'./runs/{name}/logs')
    for image, queries, mask in pbar:
        image = image.to(device)
        queries = queries.to(device)
        mask = mask.to(device)
        masked_indices = image * ~mask
        logits = model(masked_indices, mask=mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), (image * mask).view(-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f'loss: {loss.item():.4f}')

        tb_writer.add_scalar('loss', loss.item(), step)
        if step % 100 == 0:
            image = image[0]
            mask = mask[0]
            logits = logits[0]

            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated_indices = (image * ~mask + ix * mask).reshape(16, 16)
            masked_indices = (image * ~mask) + (torch.randint(50, 55, ix.shape).to(device) * mask)
            #generated_indices = ix.reshape(64, 64).long()

            unmasked_image = generate_from_indices(generated_indices, vqgan_model)
            masked_image = generate_from_indices(masked_indices.long().view(16, 16), vqgan_model)
            original_image = generate_from_indices(image.long().view(16, 16), vqgan_model)
            tb_writer.add_image('train/sample', torchvision.utils.make_grid([original_image, masked_image, unmasked_image]), step)

            #tb_writer.add_image('train/original', , step)
        step = step + 1
        if step % 1000 == 0:
            torch.save({
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'step': step,
            },
                f'./runs/{name}/gpt_model_checkpoint_{step}_.pt')


@torch.no_grad()
def generate_from_indices(indices, vqgan_model):
    indices = indices.unsqueeze(0)
    generated_latents = vqgan_model.vq.embedding(indices).permute(0, 3, 1, 2)

    reconstructed = vqgan_model.decode(generated_latents).squeeze(0)
    reconstructed = torch.clamp(reconstructed, -1, 1)
    reconstructed = (reconstructed + 1) / 2

    return reconstructed


if __name__ == '__main__':
    main()