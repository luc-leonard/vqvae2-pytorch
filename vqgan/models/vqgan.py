import torch
from torch import nn
from omegaconf import OmegaConf
from vector_quantize_pytorch import VectorQuantize

# from vqvae.vqvae import ResBlock
import torchvision.transforms as TF
import torch.nn.functional as F

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x



class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_channels, num_heads=1, batch_first=True)



    def forward(self, x):
        x_ = x

        b, c, h, w = x_.shape
        x_ = x_.reshape(b, c, h*w).permute(0, 2, 1)

        x_ = self.attn(x_, x_, x_)[0]
        x_ = x_.permute(0, 2, 1).reshape(b, c, h, w)
        return x+x_


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class Encoder2D(nn.Module):
    def __init__(self, resolution, in_channels, channels, z_channels, channel_multiplier, num_res_blocks, resolution_attention, **kwargs):
        super(Encoder2D, self).__init__()

        self.num_layers = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        in_ch_mult = (1,) + tuple(channel_multiplier)

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()

        current_resolution = resolution
        for level in range(self.num_layers):
            print(f'resolution for layer {level}: {current_resolution}')
            block_in = channels * in_ch_mult[level]
            block_out = channels * channel_multiplier[level]
            block = nn.ModuleList()
            attention = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out))
                block_in = block_out
                if current_resolution in resolution_attention:
                    attention.append(AttnBlock(block_out))
            down = nn.Module()
            down.block = block
            down.attention = attention
            if level != self.num_layers - 1:
                down.downsample = Downsample(in_channels=block_out,with_conv=False)
                current_resolution //= 2
            self.down.append(down)

        block_in = channels * channel_multiplier[-1]

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attention) > 0:
                    h = self.down[i_level].attention[i_block](h)
                hs.append(h)
            if i_level != self.num_layers-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder2D(nn.Module):
    def __init__(self, *, resolution, out_channels, channels, z_channels, channel_multiplier, num_res_blocks, resolution_attention, **kwargs):
        super(Decoder2D, self).__init__()

        self.num_layers = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks


        block_in = channels * channel_multiplier[-1]

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()

        curr_res = resolution // 2 ** (self.num_layers - 1)
        for i_level in reversed(range(self.num_layers)):
            print(f'resolution for layer {i_level}: {curr_res}')
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out))
                block_in = block_out
                if curr_res in resolution_attention:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attention = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attention) > 0:
                    h = self.up[i_level].attention[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


def make_encoder(dimension, **config):
    if dimension == 2:
        return Encoder2D(**config)

def make_decoder(dimension, **config):
    if dimension == 2:
        return Decoder2D(**config)


class VQModel(nn.Module):
    def __init__(self, codebook, encoder_decoder):
        super(VQModel, self).__init__()
        self.encoder = make_encoder(**encoder_decoder)
        self.quant_conv = nn.Conv2d(encoder_decoder.z_channels, codebook.dim, kernel_size=1)
        self.vq = VectorQuantize(dim=codebook.dim, codebook_size=codebook.n_embeds)
        self.post_quant_conv = nn.Conv2d(codebook.dim, encoder_decoder.z_channels, kernel_size=1)
        self.decoder = make_decoder(**encoder_decoder)

    def forward(self, x):
        encoded = self.quant_conv(self.encoder(x)).permute(0, 2, 3, 1)
        quantized, indices, commit_loss = self.vq(encoded)
        decoded = self.decoder(self.post_quant_conv(quantized.permute(0, 3, 1, 2)))
        return commit_loss, decoded

    def encode(self, x):
        encoded = self.quant_conv(self.encoder(x))
        quantized, indices, commit_loss = self.vq(encoded.permute(0, 2, 3, 1))
        return quantized, indices, commit_loss

    def decode(self, quantized):
        return self.decoder(self.post_quant_conv(quantized.permute(0, 3, 1, 2)))


def make_model_from_config(yaml_config_file):
    config = OmegaConf.load(yaml_config_file)
    return VQModel(**config.model.params)