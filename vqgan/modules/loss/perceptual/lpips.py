from torch import nn
import lpips

class LpipsLoss(nn.Module):
    def __init__(self, perceptual_model='vgg', device='cuda'):
        super(LpipsLoss, self).__init__()
        self.iner_loss = lpips.LPIPS(net=perceptual_model).to(device)

    def forward(self, reconstructed, target):
        return self.iner_loss(reconstructed, target)
