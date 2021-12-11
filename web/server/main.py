import flask
import torch
from PIL import Image
from flask import request
from omegaconf import OmegaConf
from torch import embedding
from torchvision.utils import make_grid

from vqgan.models import vqgan
from vqgan.models.vqgan import VQModel
import torchvision.transforms as TF

app = flask.Flask(__name__)


device = 'cpu'
config = OmegaConf.load('../../vqgan/config/ffhq_)
torch.set_printoptions(profile="full")


def preprocess_image(image):
    transforms = TF.Compose([TF.Resize((256, 256)), TF.ToTensor()])
    image_tensor = transforms(image).unsqueeze(0).to(device)
    return image_tensor * 2 - 1 # 0-1 to -1-1


def get_latent(image):
    image = preprocess_image(image)
    latent_indices = vqgan_model.encode(image)[1]

    return latent_indices

@app.post('/image')
def get_completes():
    image = Image.open(request.files.get('image').stream)
    latent_indices = get_latent(image)
    print(latent_indices.shape)
    latents = vqgan_model.vq.embedding(latent_indices).permute(0, 3, 1, 2)

    reconstructed = torch.clamp(vqgan_model.decode(latents), -1, 1)
    reconstructed = (reconstructed + 1) / 2
    TF.ToPILImage()(reconstructed).show()
