import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as TF

class MyImageFolderDataset(Dataset):
    def __init__(self,
                 data_dir,
                 extensions=['.jpg', '.jpeg', '.png'],
                 resize=(512, 512), **ignored):
        self.files = []
        for extension in extensions:
            self.files.extend(glob.glob(data_dir + '/**/*' + extension))
            self.files.extend(glob.glob(data_dir + '/*' + extension))
        self.transform = TF.Compose([TF.Resize(resize), TF.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(Image.open(self.files[index]).convert('RGB')), torch.tensor(0)
