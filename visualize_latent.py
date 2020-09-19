from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os
import datetime
import torch.nn.init as init
import cv2
import glob
import csv
import pandas as pd
import itertools

parser = argparse.ArgumentParser(description='data visualize')
parser.add_argument('--path', type=str, metavar='path', help='(default: none)')
args = parser.parse_args()


list = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6','z7','z8','z9','z10','path']
i = 0

df = pd.read_csv(args.path, header=0, index_col=0)
# print(df)

for i in range(10):
    df = df.sort_values(by=list[i])
    j = 0
    list_img = []
    # for j in range(10):
    for j in range(len(df)):
        # path = df.iloc[int(j/9.0*(len(df)-1)),10]
        path = df.iloc[j, 10]
        img = cv2.imread(path)
        img = img.transpose((2, 0, 1)) / 255.
        data = torch.from_numpy(img.astype(np.float32)).clone()
        list_img.append(data)
    # print(list_img)

    sample = torch.cat(list_img)
    # save_image(sample.view(10 , 3, 28, 28),
    #            'results/' + args.start_time + list[i] + '.png', nrow=10)
    # save_image(sample.view(10 , 3, 28, 28),
    save_image(sample.view(len(df) , 3, 28, 28),
               list[i] + '.png', nrow=20)

    # save_image(sample.view(args.latent_size * 6, 3, 28, 28),
    #            'results/' + args.start_time + '/images/sample/' + str(epoch) + '_z' + str(
    #                z + 1) + '.png', nrow=args.latent_size)
        # print(df.iloc[int(j/9.0*(len(df)-1)),i])
