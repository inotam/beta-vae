from __future__ import print_function
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import pickle
import os
import datetime

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=float, default=4, metavar='B', help='beta parameter for KL-term in original beta-VAE(default: 4)')
parser.add_argument('--latent-size', type=int, default=10, metavar='L', help='(default: 20)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#print(args)
#print(parser)
dict = {'hyper-parameter':args}
with open('./test.pickle','wb') as f:
    pickle.dump(dict, f)

with open('./test.pickle','rb') as f:
    x = pickle.load(f)
    print(x)

# x.exists("./output/v20/v%d" % i)):
# x
#
# os.mkdir("./output/v20/v%d" % i)
# os.mkdir("./output/v20/v%d/pretrain" % i)
# os.mkdir("./output/v20/v%d/pretrain/train" % i)
# os.mkdir("./output/v20/v%d/pretrain/test" % i)
# os.mkdir("./output/v20/v%d/perc_train" % i)
# os.mkdir("./output/v20/v%d/perc_train/train" % i)
# os.mkdir("./output/v20/v%d/perc_train/test" % i)
#
# output_dir = ("./output/v20/v%d" % i)

now = datetime.datetime.now()
os.mkdir("./results/"+now.strftime('%Y%m%d-%H:%M:%S')+'/')

