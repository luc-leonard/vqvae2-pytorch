from torch import nn
import torch

from latent_dffusion.model.modules.embeddings import FourierFeatures
from latent_dffusion.model.modules.unet_layers import UNetLayer


class UNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, base_hidden_channel=128, n_layers=4,
            chan_multiplier=[], inner_layers=[], attention_layers=[],
    ):
        super().__init__()
        self.input_projection = UNetLayer(
            in_channels, base_hidden_channel, inner_layers=3, downsample=False
        )

        down_layers = []
        up_layers = []

        for level in range(n_layers - 1):
            layer = UNetLayer(
                base_hidden_channel * chan_multiplier[level],
                base_hidden_channel * chan_multiplier[level + 1],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=level > 0,
            )
            down_layers.append(layer)

        for level in reversed(range(n_layers - 1)):
            layer = UNetLayer(
                base_hidden_channel * chan_multiplier[level + 1] * 2,
                base_hidden_channel * chan_multiplier[level],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                upsample=level > 0,
            )
            up_layers.append(layer)

        self.down = nn.ModuleList(down_layers)
        self.up = nn.ModuleList(up_layers)

        self.output_projection = UNetLayer(
            base_hidden_channel * chan_multiplier[0],
            out_channels,
            inner_layers=3,
            upsample=False,
            is_last=True,
        )

        with torch.no_grad():
            for param in self.parameters():
                param *= 0.5 ** 0.5

    def forward(self, x):
        x = self.input_projection(x)
        skips = []
        for down in self.down:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.up, skips[::-1]):
            x = up(torch.cat([x, skip], dim=1))

        x = self.output_projection(x)
        return x


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionModel(nn.Module):
    def __init__(
        self,
        timestep_embed=16,
        size=None,
        num_classes=10,
        in_channels=1,
        out_channels=1,
        base_hidden_channels=32,
        n_layers=2,
        chan_multiplier=[],
        inner_layers=[],
        attention_layers=[]
    ):
        super().__init__()
        self.size = size
        self.in_channel = in_channels

        self.timestep_embed = FourierFeatures(1, timestep_embed)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, timestep_embed)
        else:
            self.class_embed = None

        self.unet = UNet(
            in_channels=in_channels + timestep_embed,
            out_channels=out_channels,
            base_hidden_channel=base_hidden_channels,
            n_layers=n_layers,
            chan_multiplier=chan_multiplier,
            inner_layers=inner_layers,
            attention_layers=attention_layers,
        )

    def forward(self, x, t, class_id=None):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        if self.class_embed is not None:
            class_embed = expand_to_planes(self.class_embed(class_id), x.shape)
            timestep_embed = timestep_embed + class_embed
        x = torch.cat([x, timestep_embed], dim=1)
        x = self.unet(x)
        return x
