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
                 crop_size=None,
                 resize=None, **ignored):
        self.files = []
        for extension in extensions:
            self.files.extend(glob.glob(data_dir + '/**/*' + extension))
            self.files.extend(glob.glob(data_dir + '/*' + extension))
        if resize is not None:
            rescaler = albumentations.SmallestMaxSize(max_size=resize)
        else:
            rescaler = albumentations.NoOp()
        if crop_size is not None:
            cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
        else:
            cropper = albumentations.NoOp()
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


class LatentsDataset(Dataset):
    def __init__(self, data_dir, patch_size=None):
        self.root_dir = data_dir
        files = glob.glob(data_dir + '/*.pt')
        self.size = 0
        self.files = []
        self.patch_size = patch_size
        if patch_size:
            self.cropper = albumentations.RandomCrop(height=patch_size[0], width=patch_size[1])
            self.cropper = albumentations.Compose([self.cropper],
                                                  additional_targets={"coord": "image"})

        for file in files:
            self.files.append(torch.load(file, map_location='cpu'))
            self.size += self.files[-1].shape[0]

        # none is a 'last file' that would not contain the same amount as the others
        self.order_tensors()

    def order_tensors(self):
        # puts the only file with less tensors as the last one in the `files` array
        if self.files[0].shape[0] == self.files[-1].shape[0]:
            for idx, file in enumerate(self.files):
                if file.shape[0] != self.files[0].shape[0]:
                    # this is out last file. put it at the end of the list
                    self.files = self.files[:idx] + self.files[idx + 1:] + [file]
        else:
            if self.files[0].shape[0] < self.files[-1].shape[0]:
                self.files = self.files[1:] + self.files[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            item_per_file = self.files[0].shape[0]
            file_idx = idx // item_per_file
            item_idx = idx % item_per_file
            value = self.files[file_idx][item_idx]
            if self.patch_size is not None:
                # we need to crop the image and to add the coordinates to the input
                h, w = value.shape
                coord = np.arange(h * w).reshape(h, w, 1)
                out = self.cropper(image=value.numpy(), coord=coord)

                value = torch.flatten(torch.tensor(out['image']))
                coord = torch.flatten(torch.tensor(out['coord']))
                input = torch.cat([coord, value], dim=0)[:-1]
                target = value[1:]
            else:
                value = torch.flatten(value)
                input = value[:-1]
                target = value[1:]

            return input, target
        except Exception as e:
            print(f'{idx} => [{file_idx}][{item_idx}]')
            print(self.files[file_idx].shape)
            print(idx, e)
            raise e
