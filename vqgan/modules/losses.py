import functools

import lpips
from torch import nn
import torch

from utils.utils import get_class_from_str
from vqgan.models.discriminator import NLayerDiscriminator
from vqgan.modules.utils import ActNorm
import torch.nn.functional as F

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class CombinedLosses(nn.Module):
    def __init__(self, loss_config):
        super(CombinedLosses, self).__init__()
        self.reconstruction_loss = get_class_from_str(loss_config.reconstruction.target)()
        self.reconstruction_loss_factor = loss_config.reconstruction.factor

        self.codebook_loss_factor = loss_config.codebook.factor

        self.perceptual_loss = get_class_from_str(loss_config.perceptual.target)(**loss_config.perceptual.params)
        self.perceptual_loss_factor = loss_config.perceptual.factor

        self.discriminator = nn.Module() # empty module
        if 'discriminator' in loss_config:
            self.discriminator = get_class_from_str(loss_config.discriminator.target)(**loss_config.discriminator.params)\
                .to(loss_config.discriminator.device)
            self.discriminator_factor = loss_config.discriminator.factor
            self.disc_loss = hinge_d_loss
            self.discriminator_iter_start = loss_config.discriminator.iter_start
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=4.5e-6, betas=(0.5, 0.999))

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_factor
        return d_weight


    def forward(self, codebook_loss, target, reconstructed, optimizer_idx, global_step, last_layer=None):
        r_loss = torch.abs(target.contiguous() - reconstructed.contiguous())
        p_loss = self.perceptual_loss(target.contiguous(), reconstructed.contiguous())
        nll_loss = torch.mean(r_loss + self.perceptual_loss_factor * p_loss)

        if not self.discriminator:
            loss = nll_loss + (self.codebook_loss_factor * codebook_loss.mean())

            log = {"total_loss": loss.clone().detach().mean(),
                   "codebook_loss": codebook_loss.detach().mean(),
                   "nll_loss": nll_loss.detach().mean(),
                   "r_loss": r_loss.detach().mean(),
                   "p_loss": p_loss.detach().mean(),
                   }
            return loss, log

        # if we have a GAN, things are more complicated
        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructed.contiguous())
            g_loss = -torch.mean(logits_fake)

            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            disc_factor = adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_iter_start)

            loss = nll_loss + \
                   (d_weight * disc_factor * g_loss) + \
                   (self.codebook_loss_factor * codebook_loss.mean())

            log = {"total_loss": loss.clone().detach().mean(),
                   "codebook_loss": codebook_loss.detach().mean(),
                   "nll_loss": nll_loss.detach().mean(),
                   "r_loss": r_loss.detach().mean(),
                   "p_loss": p_loss.detach().mean(),
                   "d_weight": d_weight.detach(),
                   "disc_factor": torch.tensor(disc_factor),
                   "g_loss": g_loss.detach().mean(),
                   }
            return loss, log
        else:
            logits_real = self.discriminator(target.detach())
            logits_fake = self.discriminator(reconstructed.detach())

            disc_factor = adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_iter_start)
            loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return loss, log


class LpipsLoss(nn.Module):
    def __init__(self, perceptual_model='vgg', device='cuda'):
        super(LpipsLoss, self).__init__()
        self.iner_loss = lpips.LPIPS(net=perceptual_model).to(device)

    def forward(self, reconstructed, target):
        return self.iner_loss(reconstructed, target)
