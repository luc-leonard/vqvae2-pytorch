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

    def on_step(self, type, model, x, y, predicted, logs, step):
        for loss_name, loss_value in logs.items():
            self.tb_writer.add_scalar(type + '/' + loss_name, loss_value, step)



class ImageReconstructionTensorBoardCallback:
    def __init__(self, run_dir, every, size=256):
        self.tb_writer = SummaryWriter(run_dir + '/logs/')
        self.every = every
        self.size = size

    @torch.no_grad()
    def on_step(self, type, model, x, y, model_output, losses, step):
        if step % self.every == 0:
            sample = x[0].unsqueeze(0)
            output = model_output[1][0].unsqueeze(0)
            output = torch.clamp(output, -1, 1)
            output = (output + 1) / 2 # [-1, 1] => [0, 1]
            latents, latents_indices, _ = model.encode(sample)
            latents_indices = nn.Upsample(size=(self.size, self.size))(latents_indices.unsqueeze(0).float()).long().squeeze(0)
            latents_indices = torch.stack([latents_indices, latents_indices, latents_indices]).permute(1, 0, 2, 3)
            sample = (sample + 1) / 2
            self.tb_writer.add_image(type + '/reconstruction',
                                     torchvision.utils.make_grid(torch.cat([sample, output, latents_indices]), nrow=13),
                                     step)
            self.tb_writer.add_scalar(type + '/Used codes', torch.unique(latents_indices).shape[0], step)

