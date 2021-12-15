import random
import time

import click
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from omegaconf import OmegaConf

import vqgan.models.vqgan
from gpt.gpt import GPTConfig
from gpt.utils import sample

import torchvision.transforms as TF

from vqgan.train_gpt import make_gpt

device = 'cuda'
vqgan_device = 'cuda'


def __generate_window(gpt_model, vqgan_model, window_size=16):
    c_indices_shape = torch.Size([1, 64 * 64])
    c_code_shape = torch.Size([1, 1024, 64, 64])

    z_code_shape = c_code_shape
    z_indices_shape = c_indices_shape
    z_indices = torch.ones(z_indices_shape, device=device).long()
    z_indice_shape = z_indices.shape

    cidx = torch.tensor(np.arange(64 * 64).reshape((1, 64, 64, 1))).to(device)
    print(cidx)
    idx = z_indices
    print(z_code_shape)
    idx = idx.reshape(z_code_shape[0], z_code_shape[2], z_code_shape[3])


    temperature = 1.0
    top_k = None
    update_every = 50

    start_t = time.time()
    for i in range(0, z_code_shape[2] - 0):
        if i <= 8:
            local_i = i
        elif z_code_shape[2] - i < 8:
            local_i = 16 - (z_code_shape[2] - i)
        else:
            local_i = 8
        for j in range(0, z_code_shape[3] - 0):
            if j <= 8:
                local_j = j
            elif z_code_shape[3] - j < 8:
                local_j = 16 - (z_code_shape[3] - j)
            else:
                local_j = 8

            i_start = i - local_i
            i_end = i_start + 16
            j_start = j - local_j
            j_end = j_start + 16


            cpatch = cidx[:, i_start:i_end, j_start:j_end]
            cpatch = cpatch.reshape(cpatch.shape[0], -1)

            patch = idx[:, i_start:i_end, j_start:j_end]
            patch = patch.reshape(patch.shape[0], -1)

            patch = torch.cat((cpatch, patch), dim=1)
            logits, _ = gpt_model(patch[:, :-1])
            logits = logits[:, -256:, :]
            logits = logits.reshape(z_code_shape[0], 16, 16, -1)
            logits = logits[:, local_i, local_j, :]
            logits = logits / temperature

            if top_k is not None:
                logits = gpt_model.top_k_logits(logits, top_k)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx[:, i, j] = torch.multinomial(probs, num_samples=1)

            step = i * z_code_shape[3] + j
            if step % update_every == 0 or step == z_code_shape[2] * z_code_shape[3] - 1:
                # print(idx.shape)
                # embeds = vqgan_model.vq.embedding(idx).permute(0, 3, 1, 2)
                # image = vqgan_model.decode(embeds).squeeze(0)
                # image = torch.clamp(image, -1, 1)
                # image = (image + 1) / 2
                # TF.ToPILImage()(image).show()
                print(f"Time: {time.time() - start_t} seconds")
                print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
    print(idx)
    embeds = vqgan_model.vq.embedding(idx).permute(0, 3, 1, 2)
    image = vqgan_model.decode(embeds).squeeze(0)
    image = torch.clamp(image, -1, 1)
    image = (image + 1) / 2
    return image


@click.command()
@click.option("--checkpoint-path")
@click.option("--config-path")
@click.option("--source-image", default=None)
@torch.no_grad()
def generate(checkpoint_path, config_path, source_image):
    config = OmegaConf.load(config_path)

    gpt_config = GPTConfig(**config.model.gpt.params)
    model, _, _ = make_gpt(checkpoint_path, gpt_config)
    model = model.to(device)
    model.eval()

    vqgan_model = vqgan.models.vqgan.make_model_from_config(config.model.vqgan).to(vqgan_device)
    vqgan_model.load_from_file(config.model.vqgan.checkpoint_path)
    vqgan_model.eval()
    if 'patch_size' in config.data.params:
        return __generate_window(model, vqgan_model)

    if source_image is not None:
        image = Image.open(source_image).convert('RGB').resize((256, 256))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.tensor(image).permute(2, 0, 1).to(device)

        quantized, latent_indices, _ = vqgan_model.encode(image.unsqueeze(0))
        TF.ToPILImage()((vqgan_model.decode(quantized).squeeze(0) + 1) / 2).show()
        latent_indices = torch.flatten(latent_indices.squeeze(0))
        seed = latent_indices[0:120].unsqueeze(0).repeat_interleave(8, dim=0)
    else:
        seed = torch.clamp((torch.randn((8, 1)).to(device).to(device) * 8192).long(), 0, 8190)
    print(seed.shape)
    generated = sample(model,
                      seed,
                       16 * 16 - seed.shape[-1], temperature=1.0, sample=True,).view(-1, 16, 16)
    print(generated.shape)
    generated = generated.to(vqgan_device)
    embeds = vqgan_model.vq.embedding(generated).permute(0, 3, 1, 2)
    print(embeds.shape)
    image = vqgan_model.decode(embeds).squeeze(0)
    image = torch.clamp(image, -1, 1)
    image = (image + 1) / 2
    TF.ToPILImage()(torchvision.utils.make_grid(image)).show()


if __name__ == "__main__":
    generate()



