import click
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import get_class_from_str
from vqgan.models.vqgan import make_model_from_config
from vqgan.modules.loss.vqperceptual import VQLPIPSWithDiscriminator
from vqgan.modules.losses import CombinedLosses

device = 'cuda'


def make_callbacks(config, run_dir):
    callbacks = []
    for callback_config in config.train.callbacks:
        callbacks.append(get_class_from_str(callback_config.target)(run_dir=run_dir,
                                                                    **(callback_config.params or {})))
    return callbacks


def train(name, loss_fn, callbacks, model, dataloader, optimizer, base_step, config):
    pbar = tqdm(range(config.train.total_steps))

    data_iterator = iter(dataloader)
    disc_turn = False
    for step in pbar:
        step = base_step + step
        try:
            (image, _) = next(data_iterator)
        except Exception as e:
            data_iterator = iter(dataloader)
            continue
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        optimizer.zero_grad()
        loss_fn.discriminator_opt.zero_grad()

        image = image.to(device)
        model_output = model(image)

        # generator
        loss, log = loss_fn(model_output[0], image, model_output[1], 0, step, model.get_last_layer().weight)
        loss.backward()
        optimizer.step()

        #discriminator
        loss, log2 = loss_fn(model_output[0], image, model_output[1], 1, step, model.get_last_layer().weight)
        loss.backward()
        loss_fn.discriminator_opt.step()

        for callback in callbacks:
            callback.on_step(model, image, image, model_output, {**log, **log2}, step)

        if step % config.train.save_every == 0:
            torch.save({'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                        'discriminator': loss_fn.discriminator.state_dict(),
                        'discriminator_opt': loss_fn.discriminator_opt.state_dict(),
                        'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))

        pbar.set_description(str(round(loss.detach().item(), 4)))


    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))
    return step


@click.command()
@click.option('--name', default='celeba', help='Name of the run')
@click.option('--config', default='./config/celeba_vqvae.yml', help='Path to the config file')
@click.option('--resume-from', default=None, help='checkpoint')
@click.option('--epochs', default=1, help='base step')
def main(name, config, resume_from, epochs):
    config = OmegaConf.load(config)
    model = make_model_from_config(config.model).to(device)
    loss_fn = VQLPIPSWithDiscriminator(10_000)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    base_step = 0
    callbacks = make_callbacks(config, f'./runs/{name}/')
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        base_step = checkpoint['step']
        loss_fn.discriminator.load_state_dict(checkpoint['discriminator'])
        loss_fn.discriminator_opt.load_state_dict(checkpoint['discriminator_opt'])


    dataset = get_class_from_str(config.data.target)(**config.data.params)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=1)

    for _ in range(epochs):
        print('.')
        base_step = train(name, loss_fn, callbacks, model, dataloader, optimizer, base_step, config)


if __name__ == '__main__':
    main()

