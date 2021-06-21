import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from PIL import Image

class OpenImages(Dataset):

    def __init__(self):
        self.annotations = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/test-image-ids.csv'),
                                       usecols=['ImageID', 'OriginalURL'])

    def __getitem__(self, idx):
        batch = dict()
        img_info = self.annotations.loc[idx]

        batch['url'] = img_info.OriginalURL
        batch['idx'] = idx
        return batch

    def __len__(self):
        return len(self.annotations)


class UnsplashIndexing(Dataset):
    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self):
        self.root = '/mnt/nvme/dissertation/unsplash/full/imgs'
        self.fnames = os.listdir(self.root)

        self.urls = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/unsplash_full.tsv'), sep='\t',
                                usecols=['photo_id', 'photo_image_url'])
        self.urls['idx'] = self.urls.index
        self.urls = self.urls.set_index(['photo_id'])

    def __getitem__(self, idx):
        batch = dict()
        photo_id = self.fnames[idx].split('.')[0]
        batch['idx'] = int(self.urls.loc[photo_id, 'idx'])
        img = Image.open(os.path.join(self.root, self.fnames[idx]))
        img = self.transform(img)
        batch['img'] = img
        return batch

    def __len__(self):
        return len(self.fnames)


class Unsplash(Dataset):

    def __init__(self):
        self.urls = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/unsplash_full.tsv'), sep='\t',
                                usecols=['photo_image_url']).to_numpy()

    def __getitem__(self, idx):
        batch = dict()
        batch['url'] = self.urls[idx][0] + '?fm=jpg&w=400&fit=max'
        return batch

    def __len__(self):
        return len(self.urls)


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
        batch['img'] = self.transform(img)

        return batch

    def get_original(self, idx):
        img, cls = self.dataset[idx]

        return img

    def __len__(self):
        return len(self.dataset)
