import faiss
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *

from tqdm import tqdm

from argparse import ArgumentParser
from dataset import CustomDataset, UnsplashIndexing
import faiss
from time import sleep
import gc


def forweb():
    index = faiss.read_index(os.path.join(os.path.dirname(__file__), f'../indexes/pca.bin'))
    index.reset()

    ftrs = np.load('../features/25k.npy', allow_pickle=True).item()
    idx = np.fromiter(ftrs.keys(), dtype=int)
    f = np.empty(shape=(len(idx), 832 * 7 * 7), dtype=np.float32)
    for i, val in enumerate(ftrs.values()):
        f[i] = val
    del ftrs
    gc.collect()

    index.add_with_ids(f, idx)
    faiss.write_index(index, os.path.join(os.path.dirname(__file__), f'../indexes/unsplash_lite_pca.bin'))

def expand():
    index = faiss.read_index(os.path.join(os.path.dirname(__file__), f'../indexes/pca2048.bin'))

    for nn in range(3):
        ftrs = np.load(f'../features/unsplash_{nn + 1}.npy', allow_pickle=True).item()
        idx = np.fromiter(ftrs.keys(), dtype=int)
        f = np.empty(shape=(len(idx), 832 * 7 * 7), dtype=np.float32)
        for i, val in enumerate(ftrs.values()):
            f[i] = val
        del ftrs
        gc.collect()

        index.add_with_ids(f, idx)

        faiss.write_index(index, os.path.join(os.path.dirname(__file__), f'../indexes/pca2048_full.bin'))


def approach3():
    ftrs = np.load('../features/unsplash_0.npy', allow_pickle=True).item()
    idx = np.fromiter(ftrs.keys(), dtype=int)
    f = np.empty(shape=(len(idx), 832 * 7 * 7), dtype=np.float32)
    for i, val in enumerate(ftrs.values()):
        f[i] = val
    del ftrs
    gc.collect()

    D = 832 * 7 * 7

    # the IndexIVFPQ will be in 256D not 2048
    coarse_quantizer = faiss.IndexFlatL2(4096)
    sub_index = faiss.IndexIVFPQ(coarse_quantizer, 4096, 100, 16, 8)
    # also does a random rotation after the reduction (the 4th argument)
    pca_matrix = faiss.PCAMatrix(D, 4096, 0, False)

    # - the wrapping index
    index = faiss.IndexPreTransform(pca_matrix, sub_index)

    # Train
    print('training')
    index.train(f[:10000])
    print('done')

    index = faiss.IndexIDMap(index)
    index.add_with_ids(f, idx)

    faiss.write_index(index, os.path.join(os.path.dirname(__file__), f'../indexes/pca_ivfpq.bin'))


def approach2():
    ftrs = np.load('../features/25k.npy', allow_pickle=True).item()
    idx = np.fromiter(ftrs.keys(), dtype=int)
    f = np.empty(shape=(len(idx), 832 * 7 * 7), dtype=np.float32)
    for i, val in enumerate(ftrs.values()):
        f[i] = val
    del ftrs
    gc.collect()

    D = 832 * 7 * 7

    # the IndexIVFPQ will be in 256D not 2048
    sub_index = faiss.IndexFlatL2(4096)
    # also does a random rotation after the reduction (the 4th argument)
    pca_matrix = faiss.PCAMatrix(D, 4096, 0, False)

    # - the wrapping index
    index = faiss.IndexPreTransform(pca_matrix, sub_index)

    # Train
    print('training')
    index.train(f[:])
    print('done')

    index = faiss.IndexIDMap(index)
    index.add_with_ids(f, idx)

    faiss.write_index(index, os.path.join(os.path.dirname(__file__), f'../indexes/unsplash_lite_pca.bin'))


def approach1():
    ftrs = np.load('../features/unsplash_0.npy', allow_pickle=True).item()
    idx = np.fromiter(ftrs.keys(), dtype=int)
    f = np.empty(shape=(len(idx), 832 * 7 * 7), dtype=np.float32)
    for i, val in enumerate(ftrs.values()):
        f[i] = val
    del ftrs
    gc.collect()

    D = 832 * 7 * 7

    # Param of PQ
    M = 208  # The number of sub-vector. Typically this is 8, 16, 32, etc.
    nbits = 8  # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
    # Param of IVF
    nlist = 1000  # The number of cells (space partition). Typical value is sqrt(N)
    # Param of HNSW
    hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32

    # Setup
    quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)

    # Train
    print('training')
    index.train(f[:100000])
    print('done')

    index = faiss.IndexIDMap(index)
    index.add_with_ids(f, idx)

    faiss.write_index(index, os.path.join(os.path.dirname(__file__), f'../indexes/test.bin'))


if __name__ == '__main__':
    forweb()
