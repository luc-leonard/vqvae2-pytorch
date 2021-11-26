from torch import nn
from .resnet import ResnetBlock
from .attention import AttnBlock
from .upsample import Upsample
from .utils import Normalize, nonlinearity

class MultiLayerDecoder2D(nn.Module):
    def __init__(self, *, resolution, out_channels, channels, z_channels, channel_multiplier, num_res_blocks, resolution_attention, **kwargs):
        super(MultiLayerDecoder2D, self).__init__()

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
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

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