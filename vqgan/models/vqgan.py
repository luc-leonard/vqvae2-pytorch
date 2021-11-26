
from torch import nn
from vector_quantize_pytorch import VectorQuantize

from vqgan.modules.decoders import MultiLayerDecoder2D
from vqgan.modules.encoders import MultiLayerEncoder2D


def make_encoder(config):
    if config.dimension == 2:
        return MultiLayerEncoder2D(**config)


def make_decoder(config):
    if config.dimension == 2:
        return MultiLayerDecoder2D(**config)


class VQModel(nn.Module):
    def __init__(self, codebook, encoder_decoder):
        super(VQModel, self).__init__()
        self.encoder = make_encoder(encoder_decoder)
        self.quant_conv = nn.Conv2d(encoder_decoder.z_channels, codebook.dim, kernel_size=1)
        self.vq = VectorQuantize(dim=codebook.dim, codebook_size=codebook.n_embeds)
        self.post_quant_conv = nn.Conv2d(codebook.dim, encoder_decoder.z_channels, kernel_size=1)
        self.decoder = make_decoder(encoder_decoder)

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


def make_model_from_config(config):
    return VQModel(**config.model.params)