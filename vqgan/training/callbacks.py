from abc import ABC
from torch import nn
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Callback(ABC):
    def on_step_end(self, model, x, y, losses, step):
        ...


class LossLogCallback:
    def __init__(self, run_dir):
        self.tb_writer = SummaryWriter(run_dir + '/logs/')

    def on_step(self, model, x, y, predicted, logs, step):
        for loss_name, loss_value in logs.items():
            self.tb_writer.add_scalar(loss_name, loss_value, step)


class ImageReconstructionTensorBoardCallback:
    def __init__(self, run_dir, every):
        self.tb_writer = SummaryWriter(run_dir + '/logs/')
        self.every = every

    @torch.no_grad()
    def on_step(self, model, x, y, model_output, losses, step):
        if step % self.every == 0:
            sample = x[0].unsqueeze(0)
            output = model_output[1][0].unsqueeze(0)
            latents, latents_indices, _ = model.encode(sample)
            latents_indices = nn.Upsample(size=(512, 512))(latents_indices.unsqueeze(1).float()).squeeze(1).long()

            latents_indices = torch.stack([latents_indices, latents_indices, latents_indices]).permute(1, 0, 2, 3)

            self.tb_writer.add_image('reconstruction',
                                     torchvision.utils.make_grid(torch.cat([sample, output, latents_indices]), nrow=13),
                                     step)

