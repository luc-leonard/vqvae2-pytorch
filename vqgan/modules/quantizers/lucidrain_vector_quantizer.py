from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


class LucidVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim, commitment_cost=0.25, decay=0.8):
        super(LucidVectorQuantizer, self).__init__()
        self.quantizer = VectorQuantize(dim=dim,
                                        codebook_size=codebook_size,
                                        commitment=commitment_cost,
                                        decay=decay)

    def forward(self, x):
        quantized_x, indices, commit_loss = self.quantizer(x)
        return quantized_x, commit_loss, (None, None, indices)
