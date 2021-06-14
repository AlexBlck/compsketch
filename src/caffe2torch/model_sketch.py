#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://github.com/microsoft/MMdnn
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

__sketch_dict = dict()


def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


class SketchModel(nn.Module):
    def __init__(self, weight_file, batch_size=32):
        super(SketchModel, self).__init__()
        self.batch_size = batch_size
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((104., 117., 123.), (1., 1., 1.))
        ])
        global __sketch_dict
        __sketch_dict = load_weights(weight_file)

        self.conv1_7x7_s2_s = self.__conv(2, name='conv1/7x7_s2_s', in_channels=3, out_channels=64, kernel_size=(7, 7),
                                          stride=(2, 2), groups=1, bias=True)
        self.conv2_3x3_reduce_s = self.__conv(2, name='conv2/3x3_reduce_s', in_channels=64, out_channels=64,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2_3x3_s = self.__conv(2, name='conv2/3x3_s', in_channels=64, out_channels=192, kernel_size=(3, 3),
                                       stride=(1, 1), groups=1, bias=True)
        self.inception_3a_1x1_s = self.__conv(2, name='inception_3a/1x1_s', in_channels=192, out_channels=64,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_3x3_reduce_s = self.__conv(2, name='inception_3a/3x3_reduce_s', in_channels=192,
                                                     out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_3a_5x5_reduce_s = self.__conv(2, name='inception_3a/5x5_reduce_s', in_channels=192,
                                                     out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_3a_pool_proj_s = self.__conv(2, name='inception_3a/pool_proj_s', in_channels=192,
                                                    out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_3a_3x3_s = self.__conv(2, name='inception_3a/3x3_s', in_channels=96, out_channels=128,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_3a_5x5_s = self.__conv(2, name='inception_3a/5x5_s', in_channels=16, out_channels=32,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_5x5_reduce_s = self.__conv(2, name='inception_3b/5x5_reduce_s', in_channels=256,
                                                     out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_3b_1x1_s = self.__conv(2, name='inception_3b/1x1_s', in_channels=256, out_channels=128,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3_reduce_s = self.__conv(2, name='inception_3b/3x3_reduce_s', in_channels=256,
                                                     out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_3b_pool_proj_s = self.__conv(2, name='inception_3b/pool_proj_s', in_channels=256,
                                                    out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_3b_5x5_s = self.__conv(2, name='inception_3b/5x5_s', in_channels=32, out_channels=96,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_3b_3x3_s = self.__conv(2, name='inception_3b/3x3_s', in_channels=128, out_channels=192,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_3x3_reduce_s = self.__conv(2, name='inception_4a/3x3_reduce_s', in_channels=480,
                                                     out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4a_5x5_reduce_s = self.__conv(2, name='inception_4a/5x5_reduce_s', in_channels=480,
                                                     out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4a_1x1_s = self.__conv(2, name='inception_4a/1x1_s', in_channels=480, out_channels=192,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_pool_proj_s = self.__conv(2, name='inception_4a/pool_proj_s', in_channels=480,
                                                    out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_4a_3x3_s = self.__conv(2, name='inception_4a/3x3_s', in_channels=96, out_channels=208,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4a_5x5_s = self.__conv(2, name='inception_4a/5x5_s', in_channels=16, out_channels=48,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_3x3_reduce_s = self.__conv(2, name='inception_4b/3x3_reduce_s', in_channels=512,
                                                     out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4b_5x5_reduce_s = self.__conv(2, name='inception_4b/5x5_reduce_s', in_channels=512,
                                                     out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4b_1x1_s = self.__conv(2, name='inception_4b/1x1_s', in_channels=512, out_channels=160,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_pool_proj_s = self.__conv(2, name='inception_4b/pool_proj_s', in_channels=512,
                                                    out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_4b_3x3_s = self.__conv(2, name='inception_4b/3x3_s', in_channels=112, out_channels=224,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4b_5x5_s = self.__conv(2, name='inception_4b/5x5_s', in_channels=24, out_channels=64,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_5x5_reduce_s = self.__conv(2, name='inception_4c/5x5_reduce_s', in_channels=512,
                                                     out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4c_1x1_s = self.__conv(2, name='inception_4c/1x1_s', in_channels=512, out_channels=128,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3_reduce_s = self.__conv(2, name='inception_4c/3x3_reduce_s', in_channels=512,
                                                     out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4c_pool_proj_s = self.__conv(2, name='inception_4c/pool_proj_s', in_channels=512,
                                                    out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_4c_5x5_s = self.__conv(2, name='inception_4c/5x5_s', in_channels=24, out_channels=64,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4c_3x3_s = self.__conv(2, name='inception_4c/3x3_s', in_channels=128, out_channels=256,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_5x5_reduce_s = self.__conv(2, name='inception_4d/5x5_reduce_s', in_channels=512,
                                                     out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4d_1x1_s = self.__conv(2, name='inception_4d/1x1_s', in_channels=512, out_channels=112,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3_reduce_s = self.__conv(2, name='inception_4d/3x3_reduce_s', in_channels=512,
                                                     out_channels=144, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4d_pool_proj_s = self.__conv(2, name='inception_4d/pool_proj_s', in_channels=512,
                                                    out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_4d_5x5_s = self.__conv(2, name='inception_4d/5x5_s', in_channels=32, out_channels=64,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4d_3x3_s = self.__conv(2, name='inception_4d/3x3_s', in_channels=144, out_channels=288,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_1x1_s = self.__conv(2, name='inception_4e/1x1_s', in_channels=528, out_channels=256,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_5x5_reduce_s = self.__conv(2, name='inception_4e/5x5_reduce_s', in_channels=528,
                                                     out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4e_3x3_reduce_s = self.__conv(2, name='inception_4e/3x3_reduce_s', in_channels=528,
                                                     out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_4e_pool_proj_s = self.__conv(2, name='inception_4e/pool_proj_s', in_channels=528,
                                                    out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_4e_5x5_s = self.__conv(2, name='inception_4e/5x5_s', in_channels=32, out_channels=128,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_4e_3x3_s = self.__conv(2, name='inception_4e/3x3_s', in_channels=160, out_channels=320,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_5x5_reduce_s = self.__conv(2, name='inception_5a/5x5_reduce_s', in_channels=832,
                                                     out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_5a_3x3_reduce_s = self.__conv(2, name='inception_5a/3x3_reduce_s', in_channels=832,
                                                     out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_5a_1x1_s = self.__conv(2, name='inception_5a/1x1_s', in_channels=832, out_channels=256,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_pool_proj_s = self.__conv(2, name='inception_5a/pool_proj_s', in_channels=832,
                                                    out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_5a_5x5_s = self.__conv(2, name='inception_5a/5x5_s', in_channels=32, out_channels=128,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.inception_5a_3x3_s = self.__conv(2, name='inception_5a/3x3_s', in_channels=160, out_channels=320,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_3x3_reduce_s = self.__conv(2, name='inception_5b/3x3_reduce_s', in_channels=832,
                                                     out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_5b_1x1_s = self.__conv(2, name='inception_5b/1x1_s', in_channels=832, out_channels=384,
                                              kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5_reduce_s = self.__conv(2, name='inception_5b/5x5_reduce_s', in_channels=832,
                                                     out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                     bias=True)
        self.inception_5b_pool_proj_s = self.__conv(2, name='inception_5b/pool_proj_s', in_channels=832,
                                                    out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                    bias=True)
        self.inception_5b_3x3_s = self.__conv(2, name='inception_5b/3x3_s', in_channels=192, out_channels=384,
                                              kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.inception_5b_5x5_s = self.__conv(2, name='inception_5b/5x5_s', in_channels=48, out_channels=128,
                                              kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.lowerdim_s_1 = self.__dense(name='lowerdim_s_1', in_features=1024, out_features=256, bias=True)

    def inference(self, ims):
        """
        ims is list of numpy BRG array, float 
        """
        device = next(self.parameters()).device
        x = [self.preprocess(im.astype(np.float32)) for im in ims]
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                start_, end_ = i, min(len(x), i + self.batch_size)
                batch = torch.stack(x[start_:end_]).to(device)
                out.append(self.forward(batch).cpu().numpy())
        return np.concatenate(out)

    def forward(self, x):
        conv1_7x7_s2_s_pad = F.pad(x, (3, 3, 3, 3))
        conv1_7x7_s2_s = self.conv1_7x7_s2_s(conv1_7x7_s2_s_pad)
        conv1_relu_7x7_s = F.relu(conv1_7x7_s2_s)
        pool1_3x3_s2_s_pad = F.pad(conv1_relu_7x7_s, (0, 1, 0, 1), value=float('-inf'))
        pool1_3x3_s2_s = F.max_pool2d(pool1_3x3_s2_s_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        pool1_norm1_s = F.local_response_norm(pool1_3x3_s2_s, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        conv2_3x3_reduce_s = self.conv2_3x3_reduce_s(pool1_norm1_s)
        conv2_relu_3x3_reduce_s = F.relu(conv2_3x3_reduce_s)
        conv2_3x3_s_pad = F.pad(conv2_relu_3x3_reduce_s, (1, 1, 1, 1))
        conv2_3x3_s = self.conv2_3x3_s(conv2_3x3_s_pad)
        conv2_relu_3x3_s = F.relu(conv2_3x3_s)
        conv2_norm2_s = F.local_response_norm(conv2_relu_3x3_s, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        pool2_3x3_s2_s_pad = F.pad(conv2_norm2_s, (0, 1, 0, 1), value=float('-inf'))
        pool2_3x3_s2_s = F.max_pool2d(pool2_3x3_s2_s_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_3a_pool_s_pad = F.pad(pool2_3x3_s2_s, (1, 1, 1, 1), value=float('-inf'))
        inception_3a_pool_s = F.max_pool2d(inception_3a_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_3a_1x1_s = self.inception_3a_1x1_s(pool2_3x3_s2_s)
        inception_3a_3x3_reduce_s = self.inception_3a_3x3_reduce_s(pool2_3x3_s2_s)
        inception_3a_5x5_reduce_s = self.inception_3a_5x5_reduce_s(pool2_3x3_s2_s)
        inception_3a_pool_proj_s = self.inception_3a_pool_proj_s(inception_3a_pool_s)
        inception_3a_relu_1x1_s = F.relu(inception_3a_1x1_s)
        inception_3a_relu_3x3_reduce_s = F.relu(inception_3a_3x3_reduce_s)
        inception_3a_relu_5x5_reduce_s = F.relu(inception_3a_5x5_reduce_s)
        inception_3a_relu_pool_proj_s = F.relu(inception_3a_pool_proj_s)
        inception_3a_3x3_s_pad = F.pad(inception_3a_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_3a_3x3_s = self.inception_3a_3x3_s(inception_3a_3x3_s_pad)
        inception_3a_5x5_s_pad = F.pad(inception_3a_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_3a_5x5_s = self.inception_3a_5x5_s(inception_3a_5x5_s_pad)
        inception_3a_relu_3x3_s = F.relu(inception_3a_3x3_s)
        inception_3a_relu_5x5_s = F.relu(inception_3a_5x5_s)
        inception_3a_output_s = torch.cat(
            (inception_3a_relu_1x1_s, inception_3a_relu_3x3_s, inception_3a_relu_5x5_s, inception_3a_relu_pool_proj_s),
            1)
        inception_3b_5x5_reduce_s = self.inception_3b_5x5_reduce_s(inception_3a_output_s)
        inception_3b_1x1_s = self.inception_3b_1x1_s(inception_3a_output_s)
        inception_3b_3x3_reduce_s = self.inception_3b_3x3_reduce_s(inception_3a_output_s)
        inception_3b_pool_s_pad = F.pad(inception_3a_output_s, (1, 1, 1, 1), value=float('-inf'))
        inception_3b_pool_s = F.max_pool2d(inception_3b_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_3b_relu_5x5_reduce_s = F.relu(inception_3b_5x5_reduce_s)
        inception_3b_relu_1x1_s = F.relu(inception_3b_1x1_s)
        inception_3b_relu_3x3_reduce_s = F.relu(inception_3b_3x3_reduce_s)
        inception_3b_pool_proj_s = self.inception_3b_pool_proj_s(inception_3b_pool_s)
        inception_3b_5x5_s_pad = F.pad(inception_3b_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_3b_5x5_s = self.inception_3b_5x5_s(inception_3b_5x5_s_pad)
        inception_3b_3x3_s_pad = F.pad(inception_3b_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_3b_3x3_s = self.inception_3b_3x3_s(inception_3b_3x3_s_pad)
        inception_3b_relu_pool_proj_s = F.relu(inception_3b_pool_proj_s)
        inception_3b_relu_5x5_s = F.relu(inception_3b_5x5_s)
        inception_3b_relu_3x3_s = F.relu(inception_3b_3x3_s)
        inception_3b_output_s = torch.cat(
            (inception_3b_relu_1x1_s, inception_3b_relu_3x3_s, inception_3b_relu_5x5_s, inception_3b_relu_pool_proj_s),
            1)
        pool3_3x3_s2_s_pad = F.pad(inception_3b_output_s, (0, 1, 0, 1), value=float('-inf'))
        pool3_3x3_s2_s = F.max_pool2d(pool3_3x3_s2_s_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_4a_3x3_reduce_s = self.inception_4a_3x3_reduce_s(pool3_3x3_s2_s)
        inception_4a_5x5_reduce_s = self.inception_4a_5x5_reduce_s(pool3_3x3_s2_s)
        inception_4a_1x1_s = self.inception_4a_1x1_s(pool3_3x3_s2_s)
        inception_4a_pool_s_pad = F.pad(pool3_3x3_s2_s, (1, 1, 1, 1), value=float('-inf'))
        inception_4a_pool_s = F.max_pool2d(inception_4a_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_4a_relu_3x3_reduce_s = F.relu(inception_4a_3x3_reduce_s)
        inception_4a_relu_5x5_reduce_s = F.relu(inception_4a_5x5_reduce_s)
        inception_4a_relu_1x1_s = F.relu(inception_4a_1x1_s)
        inception_4a_pool_proj_s = self.inception_4a_pool_proj_s(inception_4a_pool_s)
        inception_4a_3x3_s_pad = F.pad(inception_4a_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_4a_3x3_s = self.inception_4a_3x3_s(inception_4a_3x3_s_pad)
        inception_4a_5x5_s_pad = F.pad(inception_4a_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_4a_5x5_s = self.inception_4a_5x5_s(inception_4a_5x5_s_pad)
        inception_4a_relu_pool_proj_s = F.relu(inception_4a_pool_proj_s)
        inception_4a_relu_3x3_s = F.relu(inception_4a_3x3_s)
        inception_4a_relu_5x5_s = F.relu(inception_4a_5x5_s)
        inception_4a_output_s = torch.cat(
            (inception_4a_relu_1x1_s, inception_4a_relu_3x3_s, inception_4a_relu_5x5_s, inception_4a_relu_pool_proj_s),
            1)
        inception_4b_3x3_reduce_s = self.inception_4b_3x3_reduce_s(inception_4a_output_s)
        inception_4b_5x5_reduce_s = self.inception_4b_5x5_reduce_s(inception_4a_output_s)
        inception_4b_pool_s_pad = F.pad(inception_4a_output_s, (1, 1, 1, 1), value=float('-inf'))
        inception_4b_pool_s = F.max_pool2d(inception_4b_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_4b_1x1_s = self.inception_4b_1x1_s(inception_4a_output_s)
        inception_4b_relu_3x3_reduce_s = F.relu(inception_4b_3x3_reduce_s)
        inception_4b_relu_5x5_reduce_s = F.relu(inception_4b_5x5_reduce_s)
        inception_4b_pool_proj_s = self.inception_4b_pool_proj_s(inception_4b_pool_s)
        inception_4b_relu_1x1_s = F.relu(inception_4b_1x1_s)
        inception_4b_3x3_s_pad = F.pad(inception_4b_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_4b_3x3_s = self.inception_4b_3x3_s(inception_4b_3x3_s_pad)
        inception_4b_5x5_s_pad = F.pad(inception_4b_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_4b_5x5_s = self.inception_4b_5x5_s(inception_4b_5x5_s_pad)
        inception_4b_relu_pool_proj_s = F.relu(inception_4b_pool_proj_s)
        inception_4b_relu_3x3_s = F.relu(inception_4b_3x3_s)
        inception_4b_relu_5x5_s = F.relu(inception_4b_5x5_s)
        inception_4b_output_s = torch.cat(
            (inception_4b_relu_1x1_s, inception_4b_relu_3x3_s, inception_4b_relu_5x5_s, inception_4b_relu_pool_proj_s),
            1)
        inception_4c_5x5_reduce_s = self.inception_4c_5x5_reduce_s(inception_4b_output_s)
        inception_4c_pool_s_pad = F.pad(inception_4b_output_s, (1, 1, 1, 1), value=float('-inf'))
        inception_4c_pool_s = F.max_pool2d(inception_4c_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_4c_1x1_s = self.inception_4c_1x1_s(inception_4b_output_s)
        inception_4c_3x3_reduce_s = self.inception_4c_3x3_reduce_s(inception_4b_output_s)
        inception_4c_relu_5x5_reduce_s = F.relu(inception_4c_5x5_reduce_s)
        inception_4c_pool_proj_s = self.inception_4c_pool_proj_s(inception_4c_pool_s)
        inception_4c_relu_1x1_s = F.relu(inception_4c_1x1_s)
        inception_4c_relu_3x3_reduce_s = F.relu(inception_4c_3x3_reduce_s)
        inception_4c_5x5_s_pad = F.pad(inception_4c_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_4c_5x5_s = self.inception_4c_5x5_s(inception_4c_5x5_s_pad)
        inception_4c_relu_pool_proj_s = F.relu(inception_4c_pool_proj_s)
        inception_4c_3x3_s_pad = F.pad(inception_4c_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_4c_3x3_s = self.inception_4c_3x3_s(inception_4c_3x3_s_pad)
        inception_4c_relu_5x5_s = F.relu(inception_4c_5x5_s)
        inception_4c_relu_3x3_s = F.relu(inception_4c_3x3_s)
        inception_4c_output_s = torch.cat(
            (inception_4c_relu_1x1_s, inception_4c_relu_3x3_s, inception_4c_relu_5x5_s, inception_4c_relu_pool_proj_s),
            1)
        inception_4d_5x5_reduce_s = self.inception_4d_5x5_reduce_s(inception_4c_output_s)
        inception_4d_1x1_s = self.inception_4d_1x1_s(inception_4c_output_s)
        inception_4d_3x3_reduce_s = self.inception_4d_3x3_reduce_s(inception_4c_output_s)
        inception_4d_pool_s_pad = F.pad(inception_4c_output_s, (1, 1, 1, 1), value=float('-inf'))
        inception_4d_pool_s = F.max_pool2d(inception_4d_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_4d_relu_5x5_reduce_s = F.relu(inception_4d_5x5_reduce_s)
        inception_4d_relu_1x1_s = F.relu(inception_4d_1x1_s)
        inception_4d_relu_3x3_reduce_s = F.relu(inception_4d_3x3_reduce_s)
        inception_4d_pool_proj_s = self.inception_4d_pool_proj_s(inception_4d_pool_s)
        inception_4d_5x5_s_pad = F.pad(inception_4d_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_4d_5x5_s = self.inception_4d_5x5_s(inception_4d_5x5_s_pad)
        inception_4d_3x3_s_pad = F.pad(inception_4d_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_4d_3x3_s = self.inception_4d_3x3_s(inception_4d_3x3_s_pad)
        inception_4d_relu_pool_proj_s = F.relu(inception_4d_pool_proj_s)
        inception_4d_relu_5x5_s = F.relu(inception_4d_5x5_s)
        inception_4d_relu_3x3_s = F.relu(inception_4d_3x3_s)
        inception_4d_output_s = torch.cat(
            (inception_4d_relu_1x1_s, inception_4d_relu_3x3_s, inception_4d_relu_5x5_s, inception_4d_relu_pool_proj_s),
            1)
        inception_4e_1x1_s = self.inception_4e_1x1_s(inception_4d_output_s)
        inception_4e_5x5_reduce_s = self.inception_4e_5x5_reduce_s(inception_4d_output_s)
        inception_4e_3x3_reduce_s = self.inception_4e_3x3_reduce_s(inception_4d_output_s)
        inception_4e_pool_s_pad = F.pad(inception_4d_output_s, (1, 1, 1, 1), value=float('-inf'))
        inception_4e_pool_s = F.max_pool2d(inception_4e_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_4e_relu_1x1_s = F.relu(inception_4e_1x1_s)
        inception_4e_relu_5x5_reduce_s = F.relu(inception_4e_5x5_reduce_s)
        inception_4e_relu_3x3_reduce_s = F.relu(inception_4e_3x3_reduce_s)
        inception_4e_pool_proj_s = self.inception_4e_pool_proj_s(inception_4e_pool_s)
        inception_4e_5x5_s_pad = F.pad(inception_4e_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_4e_5x5_s = self.inception_4e_5x5_s(inception_4e_5x5_s_pad)
        inception_4e_3x3_s_pad = F.pad(inception_4e_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_4e_3x3_s = self.inception_4e_3x3_s(inception_4e_3x3_s_pad)
        inception_4e_relu_pool_proj_s = F.relu(inception_4e_pool_proj_s)
        inception_4e_relu_5x5_s = F.relu(inception_4e_5x5_s)
        inception_4e_relu_3x3_s = F.relu(inception_4e_3x3_s)
        inception_4e_output_s = torch.cat(
            (inception_4e_relu_1x1_s, inception_4e_relu_3x3_s, inception_4e_relu_5x5_s, inception_4e_relu_pool_proj_s),
            1)
        pool4_3x3_s2_s_pad = F.pad(inception_4e_output_s, (0, 1, 0, 1), value=float('-inf'))
        pool4_3x3_s2_s = F.max_pool2d(pool4_3x3_s2_s_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        inception_5a_5x5_reduce_s = self.inception_5a_5x5_reduce_s(pool4_3x3_s2_s)
        inception_5a_pool_s_pad = F.pad(pool4_3x3_s2_s, (1, 1, 1, 1), value=float('-inf'))
        inception_5a_pool_s = F.max_pool2d(inception_5a_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_5a_3x3_reduce_s = self.inception_5a_3x3_reduce_s(pool4_3x3_s2_s)
        inception_5a_1x1_s = self.inception_5a_1x1_s(pool4_3x3_s2_s)
        inception_5a_relu_5x5_reduce_s = F.relu(inception_5a_5x5_reduce_s)
        inception_5a_pool_proj_s = self.inception_5a_pool_proj_s(inception_5a_pool_s)
        inception_5a_relu_3x3_reduce_s = F.relu(inception_5a_3x3_reduce_s)
        inception_5a_relu_1x1_s = F.relu(inception_5a_1x1_s)
        inception_5a_5x5_s_pad = F.pad(inception_5a_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_5a_5x5_s = self.inception_5a_5x5_s(inception_5a_5x5_s_pad)
        inception_5a_relu_pool_proj_s = F.relu(inception_5a_pool_proj_s)
        inception_5a_3x3_s_pad = F.pad(inception_5a_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_5a_3x3_s = self.inception_5a_3x3_s(inception_5a_3x3_s_pad)
        inception_5a_relu_5x5_s = F.relu(inception_5a_5x5_s)
        inception_5a_relu_3x3_s = F.relu(inception_5a_3x3_s)
        inception_5a_output_s = torch.cat(
            (inception_5a_relu_1x1_s, inception_5a_relu_3x3_s, inception_5a_relu_5x5_s, inception_5a_relu_pool_proj_s),
            1)
        inception_5b_pool_s_pad = F.pad(inception_5a_output_s, (1, 1, 1, 1), value=float('-inf'))
        inception_5b_pool_s = F.max_pool2d(inception_5b_pool_s_pad, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                           ceil_mode=False)
        inception_5b_3x3_reduce_s = self.inception_5b_3x3_reduce_s(inception_5a_output_s)
        inception_5b_1x1_s = self.inception_5b_1x1_s(inception_5a_output_s)
        inception_5b_5x5_reduce_s = self.inception_5b_5x5_reduce_s(inception_5a_output_s)
        inception_5b_pool_proj_s = self.inception_5b_pool_proj_s(inception_5b_pool_s)
        inception_5b_relu_3x3_reduce_s = F.relu(inception_5b_3x3_reduce_s)
        inception_5b_relu_1x1_s = F.relu(inception_5b_1x1_s)
        inception_5b_relu_5x5_reduce_s = F.relu(inception_5b_5x5_reduce_s)
        inception_5b_relu_pool_proj_s = F.relu(inception_5b_pool_proj_s)
        inception_5b_3x3_s_pad = F.pad(inception_5b_relu_3x3_reduce_s, (1, 1, 1, 1))
        inception_5b_3x3_s = self.inception_5b_3x3_s(inception_5b_3x3_s_pad)
        inception_5b_5x5_s_pad = F.pad(inception_5b_relu_5x5_reduce_s, (2, 2, 2, 2))
        inception_5b_5x5_s = self.inception_5b_5x5_s(inception_5b_5x5_s_pad)
        inception_5b_relu_3x3_s = F.relu(inception_5b_3x3_s)
        inception_5b_relu_5x5_s = F.relu(inception_5b_5x5_s)
        inception_5b_output_s = torch.cat(
            (inception_5b_relu_1x1_s, inception_5b_relu_3x3_s, inception_5b_relu_5x5_s, inception_5b_relu_pool_proj_s),
            1)
        pool5_7x7_s1_s = F.avg_pool2d(inception_5b_output_s, kernel_size=(7, 7), stride=(1, 1), padding=(0,),
                                      ceil_mode=False)
        lowerdim_s_0 = pool5_7x7_s1_s.view(pool5_7x7_s1_s.size(0), -1)
        lowerdim_s_1 = self.lowerdim_s_1(lowerdim_s_0)
        out = F.normalize(lowerdim_s_1)
        return out

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__sketch_dict[name]['weights']))
        if 'bias' in __sketch_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__sketch_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__sketch_dict[name]['weights']))
        if 'bias' in __sketch_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__sketch_dict[name]['bias']))
        return layer
