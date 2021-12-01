import flask
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import embedding
from torchvision.utils import make_grid

from vqgan.models import vqgan
from vqgan.models.vqgan import VQModel
import torchvision.transforms as TF

app = flask.Flask(__name__)


device = 'cpu'
config = OmegaConf.load('../../vqgan/config/ffhq_small_gan_f8.yml')
print(config)
vqgan_model: VQModel = vqgan.make_model_from_config(config.model).to(device)
checkpoint = torch.load('../../vqgan/runs/ffhq_256_gan/vqgan_273000.pt', map_location='cpu')['model']
vqgan_model.load_state_dict(checkpoint)
vqgan_model.eval()
torch.set_printoptions(profile="full")

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def preprocess_image(image_path):
    transforms = TF.Compose([TF.Resize((256, 256)), TF.ToTensor()])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms(image).unsqueeze(0).to(device)
    return image_tensor * 2 - 1 # 0-1 to -1-1


@app.get('/latents')
def get_latent():
    #image_path = '/media/lleonard/big_slow_disk/datasets/ffhq/images1024x1024/00000/00009.png'
    image_path = '/home/lleonard/Pictures/Star_Wars__Battlefront_II_2017_Screenshot_2018.11.14_-_18.31.03.14-min.png'
    image = preprocess_image(image_path)
    latent_indices = vqgan_model.encode(image)[1]
    latent_indices[0, 0, 0] = 0
    latent_indices[0, 0, 1] = 0
    latent_indices[0, 1, 0] = 0

    return latent_indices

@app.post('/')
def get_image():
    image_path = '/media/lleonard/big_slow_disk/datasets/ffhq/images1024x1024/00000/00009.png'
    latent_indices = get_latent()
    print(sizeof_fmt(latent_indices.element_size() * latent_indices.nelement()))
    print(latent_indices.shape)
    latent_indices_cpy = torch.tensor(latent_indices)
    reconstructeds = []
    for i in range(1):
        latent_indices = torch.tensor(latent_indices_cpy)
        #latent_indices[0, i, i] = i
        latents = vqgan_model.vq.embedding(latent_indices).permute(0, 3, 1, 2)

        reconstructed = torch.clamp(vqgan_model.decode(latents), -1, 1)
        reconstructed = (reconstructed + 1) / 2
        reconstructeds.append(reconstructed.squeeze(0))
    TF.ToPILImage()(make_grid(reconstructeds)).show()


get_image()