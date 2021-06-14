import faiss
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *

from tqdm import tqdm

from argparse import ArgumentParser
from dataset import CustomDataset


def train_faiss(args):
    # Feature extractor
    fe = models.googlenet(pretrained=True, aux_logits=False)
    fe = nn.Sequential(*list(fe.children())[:-4])
    fe = fe.eval().to(args.device)

    # Dataset
    dataset = CustomDataset(args.root)
    loader = DataLoader(dataset=dataset, batch_size=args.bs, shuffle=False)

    model = faiss.IndexFlatL2(832 * 7 * 7)
    model = faiss.IndexIDMap(model)

    print('Building index')
    for batch in tqdm(loader):
        imgs = batch['img']
        idx = batch['idx'].detach().cpu().numpy().astype(int)
        features = fe(imgs.to(args.device)).flatten(1)
        features = F.normalize(features)
        features = features.detach().cpu().numpy().astype(np.float32)
        model.add_with_ids(features, idx)

    faiss.write_index(model, os.path.join(os.path.dirname(__file__), f'../indexes/{args.name}.bin'))
    print('Done!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='GPU device id, -1 for cpu')
    parser.add_argument('--name', type=str, required=True, default='my_index', help='index filename')
    args = parser.parse_args()

    train_faiss(args)
