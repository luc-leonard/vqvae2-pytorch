from io import BytesIO
import time
from pathlib import Path

import flask
import torch
import torchvision
from PIL import Image
from flask import request, send_file
from flask_cors import CORS
from omegaconf import OmegaConf
from torch import embedding
from torchvision.utils import make_grid

from gpt.gpt import GPTConfig
from gpt.utils import sample
from vqgan.models import vqgan
from vqgan.models.vqgan import VQModel
import torchvision.transforms as TF

from vqgan.train_gpt import make_gpt

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

device = 'cuda'
config = OmegaConf.load('../../vqgan/config/ffhq_256_transformer_f16_8192.yml')
config.model.vqgan.checkpoint_path = '../../vqgan/' + config.model.vqgan.checkpoint_path

gpt_config = GPTConfig(**config.model.gpt.params)
model, _, _ = make_gpt('../../vqgan/runs/ffhq_256_f16_8192_transformer/gpt_model_checkpoint_165000.pt', gpt_config)
model = model.to(device)
model.eval()

vqgan_model = vqgan.make_model_from_config(config.model.vqgan).to(device)
vqgan_model.load_from_file(config.model.vqgan.checkpoint_path)
vqgan_model.eval()

torch.set_printoptions(profile="full")

generated_path = Path('./generated/')
generated_path.mkdir(exist_ok=True)

def preprocess_image(image):
    # align face
    transforms = TF.Compose([TF.Resize((256, 256)), TF.ToTensor()])
    image_tensor = transforms(image).unsqueeze(0).to(device)
    return image_tensor * 2 - 1 # 0-1 to -1-1


def get_latent(image):
    image = preprocess_image(image)
    latent_indices = vqgan_model.encode(image)[1]

    return torch.flatten(latent_indices)


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.get('/generated/<path:path>')
def get_generated(path):
    return send_file(generated_path / path)

@app.post('/upload')
def get_completes():
    image = Image.open(request.files.get('image').stream).convert('RGB')
    latent_indices = get_latent(image)

    seed = latent_indices[0:100].unsqueeze(0).repeat_interleave(2, dim=0)

    generated = sample(model,
                      seed,
                      256 - seed.shape[-1], temperature=1.0, sample=True, ).view(-1, 16, 16)
    print(generated.shape)
    embeds = vqgan_model.vq.embedding(generated).permute(0, 3, 1, 2)
    image = vqgan_model.decode(embeds).squeeze(0)
    image = torch.clamp(image, -1, 1)
    image = (image + 1) / 2
    image_path = generated_path / f'{int(time.time())}.jpg'
    TF.ToPILImage()(torchvision.utils.make_grid(image)).save(image_path)
    return str(image_path)


app.run()