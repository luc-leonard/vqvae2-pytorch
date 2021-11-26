from torch import nn


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