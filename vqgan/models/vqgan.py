
from torch import nn
from vector_quantize_pytorch import VectorQuantize

from utils.utils import get_class_from_str
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
        self.encoder = get_class_from_str(encoder_decoder.encoder_target)(**encoder_decoder.params)
        self.quant_conv = nn.Conv2d(encoder_decoder.params.z_channels, codebook.params.dim, kernel_size=1)

        self.vq = get_class_from_str(codebook.target)(**codebook.params)

        self.post_quant_conv = nn.Conv2d(codebook.params.dim, encoder_decoder.params.z_channels, kernel_size=1)

        self.decoder = get_class_from_str(encoder_decoder.decoder_target)(**encoder_decoder.params)

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

    def get_last_layer(self):
        return self.decoder.get_last_layer()

def make_model_from_config(config):
    return get_class_from_str(config.target)(**config.params)
