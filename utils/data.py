import glob
from PIL import Image
from torch.utils.data import Dataset


class MyImageFolderDataset(Dataset):
    def __init__(self,
                 data_dir,
                 extensions=['.jpg', '.jpeg', '.png'],
                 transform=None):
        self.files = []
        for extension in extensions:
            self.files.extend(glob.glob(data_dir + '/*' + extension))
        self.transform = transform

    def __getitem__(self, index):
        return Image.open(self.files[index]).convert('RGB'), None
