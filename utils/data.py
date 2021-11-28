import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as TF
import albumentations
import traceback

class MyImageFolderDataset(Dataset):
    def __init__(self,
                 data_dir,
                 extensions=['.jpg', '.jpeg', '.png'],
                 resize=(512, 512), **ignored):
        self.files = []
        for extension in extensions:
            self.files.extend(glob.glob(data_dir + '/**/*' + extension))
            self.files.extend(glob.glob(data_dir + '/*' + extension))
        rescaler = albumentations.SmallestMaxSize(max_size=resize)
        cropper = albumentations.CenterCrop(height=resize[0], width=resize[1])
        self.transform = albumentations.Compose([rescaler, cropper])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index]).convert('RGB')
            image = np.array(image).astype(np.uint8)
            image = self.transform(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
            return torch.tensor(image).permute(2, 0, 1), torch.tensor(0)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
