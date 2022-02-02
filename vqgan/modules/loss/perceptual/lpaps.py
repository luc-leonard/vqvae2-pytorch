import numpy as np
import torch
from lpips import NetLinLayer, normalize_tensor, spatial_average
from torch import nn
from vqgan.modules.loss.perceptual.vggish import VGGishish, vggishish16


class LPAPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vggish16 features
        self.net = vggishish16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vggishish_lpaps"):
        ckpt = './vggishish16.pt'
        print(self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False))
        print("loaded pretrained LPAPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vggishish_lpaps"):
        if name != "vggishish_lpaps":
            raise NotImplementedError
        model = cls()
        ckpt = './vggishish16.pt'
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        print(in0_input.shape, in1_input.shape)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # we are gonna use get_ckpt_path to donwload the stats as well
        # if for images we normalize on the channel dim, in spectrogram we will norm on frequency dimension
        means, stds = np.loadtxt('train_means_stds_melspec_10s_22050hz.txt', dtype=np.float32).T
        # the normalization in means and stds are given for [0, 1], but specvqgan expects [-1, 1]:
        means = 2 * means - 1
        stds = 2 * stds
        # input is expected to be (B, 1, F, T)
        self.register_buffer('shift', torch.from_numpy(means)[None, None, :, None])
        self.register_buffer('scale', torch.from_numpy(stds)[None, None, :, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

