import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class OpenImages(Dataset):

    def __init__(self):
        self.annotations = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/test-image-ids.csv'),
                                       usecols=['ImageID', 'OriginalURL'])

    def __getitem__(self, idx):
        batch = dict()
        img_info = self.annotations.loc[idx]

        batch['img_or_url'] = img_info.OriginalURL
        batch['idx'] = idx
        return batch

    def __len__(self):
        return len(self.annotations)


class CustomDataset(Dataset):
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, root):
        self.dataset = ImageFolder(root)

    def __getitem__(self, idx):
        batch = dict()
        img, cls = self.dataset[idx]
        batch['idx'] = idx
        batch['img_or_url'] = img
        batch['img'] = self.transform(img)

        return batch

    def __len__(self):
        return len(self.dataset)
