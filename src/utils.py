import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image


def box_from_array(a):
    y, x = np.where(a < 255)
    im_w = a.shape[1]
    im_h = a.shape[0]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    bbox = [xmin, ymin, xmax, ymax]

    qmask = [np.floor(bbox[0] / im_w * 31), np.floor(bbox[1] / im_h * 31),
             np.ceil(bbox[2] / im_w * 31), np.ceil(bbox[3] / im_h * 31)]
    qmask = [int(x) for x in qmask]
    qb = [qmask[1], qmask[3], qmask[0], qmask[2]]

    mb = [int(np.round(x / 30 * 7)) for x in qb]
    rb = [float(x / 30) for x in qb]
    return qb, mb, rb


def box_from_fname(fname):
    qb = [int(x) for x in fname.split('.')[0].split('_')]
    mb = [int(np.round(x / 31 * 7)) for x in qb]
    rb = [float(x / 31) for x in qb]
    return qb, mb, rb
