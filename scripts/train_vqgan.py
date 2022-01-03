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


def do_step(optimizer, loss_fn, model, image, step, optimizer_idx, callbacks):
    optimizer.zero_grad()
    model_output = model(image)
    loss, log = loss_fn(model_output[0], image, model_output[1], optimizer_idx, step, model.get_last_layer().weight)
    loss.backward()
    optimizer.step()
    for callback in callbacks:
        callback.on_step('train', model, image, image, model_output, log, step)


def epoch(name, loss_fn, callbacks, model, train_dl, optimizer, base_step, config):
    for i, (image, _) in tqdm(enumerate(train_dl), total=len(train_dl)):
        print(image.shape)
        step = base_step + i
        # if len(image.shape) == 3:
        #     image = image.unsqueeze(0)
        image = image.to(device)
        do_step(loss_fn.discriminator_opt, loss_fn, model, image, step, 1, callbacks)
        do_step(optimizer, loss_fn, model, image, step, 0, callbacks)


        if step % config.train.save_every == 0:
            torch.save({'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                        'discriminator': loss_fn.discriminator.state_dict(),
                        'discriminator_opt': loss_fn.discriminator_opt.state_dict(),
                        'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))


    torch.save({'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'discriminator': loss_fn.discriminator.state_dict(),
                'discriminator_opt': loss_fn.discriminator_opt.state_dict(),
                'step': step}, "./runs/{}/vqgan_{}.pt".format(name, step))

    print("Evaluating...")


    return step


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@click.command()
@click.option('--name', default='celeba', help='Name of the run')
@click.option('--config', default='./config/celeba_vqvae.yml', help='Path to the config file')
@click.option('--resume-from', default=None, help='checkpoint')
@click.option('--epochs', default=1, help='base step')
@click.option('--seed', default=1, help='base step')
def main(name, config, resume_from, epochs, seed):
    config = OmegaConf.load(config)
    model = make_model_from_config(config.model).to(device)
    print(model)
    lr = config.train.lr * config.train.batch_size
    loss_fn = CombinedLosses(config.loss, lr)

    optimizer = torch.optim.Adam(list(model.encoder.parameters()) +
                     list(model.decoder.parameters()) +
                     list(model.vq.parameters()) +
                     list(model.quant_conv.parameters()) +
                     list(model.post_quant_conv.parameters()),
                     lr=lr, betas=(0.5, 0.9))
    seed_everything(seed)
    base_step = 0
    callbacks = make_callbacks(config, f'./runs/{name}/')
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        base_step = checkpoint['step']
        loss_fn.discriminator.load_state_dict(checkpoint['discriminator'])
        loss_fn.discriminator_opt.load_state_dict(checkpoint['discriminator_opt'])
        del checkpoint


    dataset = get_class_from_str(config.data.target)(**config.data.params)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - len(dataset) // 10, len(dataset) // 10])
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    print(f'train data: {len(train_dataset)}')
    print(f'test data: {len(test_dataset)}')
    current_epoch = base_step // len(train_dataset)
    for _ in range(epochs):
        print(f'epoch {current_epoch}, base_step {base_step}')
        model = model.train()
        new_base_step = epoch(name, loss_fn, callbacks, model, train_dataloader, optimizer, base_step, config)
        model = model.eval()
        validate(base_step, callbacks, loss_fn, model, test_dataloader)
        base_step = new_base_step
        current_epoch = current_epoch + 1

def validate(base_step, callbacks, loss_fn, model, test_dataloader):
    for i, (image, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        step = base_step + i

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        model_output = model(image)
        _, log = loss_fn(model_output[0], image, model_output[1], 0, step, model.get_last_layer().weight)
        for callback in callbacks:
            callback.on_step('valid', model, image, image, model_output, log, step)


if __name__ == '__main__':
    main()


