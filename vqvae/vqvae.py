import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize


class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, dim_hidden, num_residual_layer, dim_residual_layer):
        super(Encoder, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, dim_hidden // 2, padding=1, kernel_size=4, stride=2),
            nn.BatchNorm2d(dim_hidden // 2),
            nn.ReLU(True),
            nn.Conv2d(dim_hidden // 2, dim_hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim_hidden),
            nn.ReLU(True),
            nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=1),
        )

        self.residual_stack = nn.Sequential(*[ResBlock(dim_hidden, dim_residual_layer) for _ in range(num_residual_layer)], nn.ReLU(True))

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.residual_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, dim_hidden, num_residual_layer, dim_residual_layer):
        super(Decoder, self).__init__()

        self.deconv_stack = nn.Sequential(
            nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(dim_hidden, dim_residual_layer) for _ in range(num_residual_layer)]),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_hidden, dim_hidden //2 , kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_hidden //2, in_channels, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, x):
        x = self.deconv_stack(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, codebook_size, num_residual_layers, dim_residual_layers, **ignore_kwargs):
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, num_residual_layers, dim_residual_layers)
        self.pre_vq_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=embedding_dim, kernel_size=1, stride=1)
        self.vq = VectorQuantize(
            dim = embedding_dim,
            codebook_size = codebook_size,     # codebook size
            decay = 0.99,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment = 0.25         # the weight on the commitment loss
        )
        self.decoder = Decoder(input_dim, embedding_dim, num_residual_layers, dim_residual_layers)

    def encode(self, x):
        z_e_x = self.pre_vq_conv(self.encoder(x))
        latents, indices, _ = self.vq(z_e_x.permute(0, 2, 3, 1))
        return latents, indices

    def decode(self, latents):
        z_q_x = latents.permute(0, 3, 1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        encoded_x = self.pre_vq_conv(self.encoder(x)).permute(0, 2, 3, 1)
        quantized_x, indices, commit_loss = self.vq(encoded_x)
        x_tilde = self.decoder(quantized_x.permute(0, 3, 1, 2))
        return commit_loss, x_tilde


def load_vqvae(config, model_path):
    data = torch.load(model_path, map_location='cpu')
    print(config)
    model = VQVAE(**config)
    model.load_state_dict(data['model'])
    return model
