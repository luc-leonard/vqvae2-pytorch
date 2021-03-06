from torch import nn

from utils.utils import get_class_from_str
from ..utils import Normalize, nonlinearity
from ..resnet import ResnetBlock, ResnetBlock1D
from ..attention import AttnBlock
from ..upsample import Downsample, Downsample1D


class WaveformEncoder(nn.Module):
    def __init__(self, samples_per_second, in_channels, channels, z_channels, channel_multiplier, num_res_blocks,
                 attention_layers_at, attention_target, **kwargs):
        super(WaveformEncoder, self).__init__()

        self.num_layers = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        in_ch_mult = (1,) + tuple(channel_multiplier)

        self.conv_in = nn.Conv1d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()

        current_block_per_second = samples_per_second
        AttnClass = get_class_from_str(attention_target)
        for level in range(self.num_layers):
            print(f'blocks per second for layer {level}: {current_block_per_second}')
            block_in = channels * in_ch_mult[level]
            block_out = channels * channel_multiplier[level]
            block = nn.ModuleList()
            attention = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in,
                                         out_channels=block_out))
                block_in = block_out
                if level in attention_layers_at:
                    print(f'attention layer at level {level}')
                    attention.append(AttnClass(block_out))
            down = nn.Module()
            down.block = block
            down.attention = attention
            if level != self.num_layers - 1:
                down.downsample = Downsample1D(in_channels=block_out, with_conv=False)
                current_block_per_second //= 2
            print(f'current block at layer {level}: {current_block_per_second}')
            self.down.append(down)

        block_in = channels * channel_multiplier[-1]

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnClass(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            print(f'level {i_level} {hs[-1].shape}')
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attention) > 0:
                    h = self.down[i_level].attention[i_block](h)
                hs.append(h)
            if i_level != self.num_layers - 1:
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


class MultiLayerEncoder2D(nn.Module):
    def __init__(self, resolution, in_channels, channels, z_channels, channel_multiplier, num_res_blocks, resolution_attention, attention_layer_cls_name = None, **kwargs):
        super(MultiLayerEncoder2D, self).__init__()

        self.num_layers = len(channel_multiplier)
        self.num_res_blocks = num_res_blocks
        in_ch_mult = (1,) + tuple(channel_multiplier)

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()

        current_resolution = resolution
        AttentionLayer = AttnBlock
        if attention_layer_cls_name is not None:
            AttentionLayer = get_class_from_str(attention_layer_cls_name)


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
                    attention.append(AttentionLayer(block_out))
            down = nn.Module()
            down.block = block
            down.attention = attention
            if level != self.num_layers - 1:
                down.downsample = Downsample(in_channels=block_out, with_conv=True)
                current_resolution //= 2
            self.down.append(down)

        block_in = channels * channel_multiplier[-1]

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttentionLayer(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def get_compression_factor(self):
        return 2 ** (self.num_layers - 1)

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

