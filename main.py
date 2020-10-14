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
import cloudpickle
import csv
import pandas as pd
import itertools
import math

now = datetime.datetime.now()
start_time =  now.strftime('%Y%m%d%H%M%S')
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
parser.add_argument('--beta', type=float, default=4., metavar='B', help='beta parameter for KL-term in original beta-VAE(default: 4)')
parser.add_argument('--latent-size', type=int, default=10, metavar='L', help='(default: 20)')
parser.add_argument('--start-time', type=str, default=start_time, metavar='ST', help='(default: today_time)')
parser.add_argument('--fmnist', type=bool, default=True, metavar='FM', help='(default: fashion_MNIST)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

os.mkdir("./results/"+args.start_time+'/')
os.mkdir("./results/"+args.start_time+'/images/')
os.mkdir("./results/"+args.start_time+'/images/test')
os.mkdir("./results/"+args.start_time+'/images/sample')

dict = {'hyper-parameter':args}

if args.fmnist:

    train_dataset = datasets.ImageFolder(
        '../data/FashionMNIST/train',
        transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]))

    test_dataset = datasets.ImageFolder(
        '../data/FashionMNIST/test',
        transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]))
    # with open('../data/train.pickle','rb') as f:
    #     train_dataset = cloudpickle.load(f)
    #
    # with open('../data/test.pickle', 'rb') as f:
    #     test_dataset = cloudpickle.load(f)

    chn_num = 1
    image_size = 28
else:
    train_dataset = datasets.ImageFolder(
            # '../data/faceless_300/train_d',
            '../data/noskin_all_v2/noskin_28/train_d',
            transforms.Compose([
                transforms.ToTensor(),
            ]))

    test_dataset = datasets.ImageFolder(
            # '../data/faceless_300/test_d',
            '../data/noskin_all_v2/noskin_28/test_d',
            transforms.Compose([
                transforms.ToTensor(),
            ]))
    chn_num = 3
    image_size = 28

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self, chn_num=1, image_size=28, latent=10):
        super(VAE, self).__init__()
        self.chn_num = chn_num
        self.image_size = image_size
        self.latent = latent

        self.fc1 = nn.Linear(image_size * image_size * chn_num, 400)
        self.fc21 = nn.Linear(400, latent)
        self.fc22 = nn.Linear(400, latent)
        self.fc3 = nn.Linear(latent, 400)
        self.fc4 = nn.Linear(400, image_size * image_size * chn_num)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.image_size * self.image_size * self.chn_num), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + args.beta * KLD, BCE, KLD

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.image_size * self.image_size * self.chn_num))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(chn_num,image_size,args.latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, image_size * image_size * chn_num), reduction='sum')
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return BCE + args.beta*KLD, BCE, KLD


def train(epoch):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        optimizer.step()

        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    if epoch % 10 == 0:
        print('====> Epoch: {} Average loss: {:.8f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        print('====> Epoch: {} Average BCE: {:.8f}'.format(
            epoch, train_bce / len(train_loader.dataset)))
        print('====> Epoch: {} Average KLD: {:.8f}'.format(
              epoch, train_kld / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    test_bce = 0
    test_kld = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            # print(0)
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            loss, bce, kld = model.loss_function(recon_batch, data, mu, logvar)
            test_loss += loss
            test_bce += bce
            test_kld += kld
            if i == 0 and (epoch % 10 == 0):
                n = min(data.size(0), 10)
                batch_size = min(data.size(0), args.batch_size)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, chn_num, image_size, image_size)[:n]])
                # recon_batch.view(1, 3, 300, 300)[:n]])
                save_image(comparison.cpu(),
                           'results/' + args.start_time + '/images/test/' + str(epoch) + '.png', nrow=n)

    #test_loss /= len(test_loader.dataset)
    if epoch % 10 ==0:
        print('====> Test set loss: {:.8f}'.format(test_loss/len(test_loader.dataset)))
        print('====> Test set BCE: {:.8f}'.format(test_bce/ len(test_loader.dataset)))
        print('====> Test set KLD: {:.8f}'.format(test_kld / len(test_loader.dataset)))

def make_db(path):
    with torch.no_grad():
        list = []
        file = glob.glob(path)
        for f in file:
            img = cv2.imread(f)
            img = img.transpose((2, 0, 1))/255.
            data = torch.from_numpy(img.astype(np.float32)).clone().to(device)

            data = data.unsqueeze(0)

            # mu, logvar = model.encode(data.contiguous().view(-1, 784 * 3))
            mu, logvar = model.encode(data.contiguous().view(-1, image_size * image_size * chn_num))
            z = model.reparameterize(mu, logvar).cpu().detach().numpy().copy()
            z = z.tolist()
            z[0].append(f)
            list.append(np.array(z[0]))
    df = pd.DataFrame(list, columns=['z1', 'z2', 'z3', 'z4', 'z5', 'z6','z7','z8','z9','z10','path'])

    return df

def latant_space_exploration(df, name):
    df.to_csv('./results/' + args.start_time + '/db_' + str(name)+'.csv')
    col_list = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6','z7','z8','z9','z10']

    if args.fmnist:
        range_obj = range(0, int(len(df)/10), 9)
    else:
        range_obj = range(len(df))

    for i in range(10):
        df = df.sort_values(by=col_list[i])
        j = 0
        list_img = []
        for j in range_obj:
            # path = df.iloc[int(j/9.0*(len(df)-1)),10]
            path = df.iloc[j, 10]

            img = cv2.imread(path)
            img = img.transpose((2, 0, 1)) / 255.
            data = torch.from_numpy(img.astype(np.float32)).clone()
            list_img.append(data)

        sample = torch.cat(list_img)
        save_image(sample.view(len(list_img), 3, image_size, image_size),
                   'results/' + args.start_time + '/' + col_list[i] + '_' + str(name) + '.png', nrow=int(math.sqrt(len(list_img))))


if __name__ == "__main__":
    latent_size = args.latent_size
    interpolation = torch.linspace(-3, 3, 10)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:
            with torch.no_grad():
                for z in range(args.latent_size):
                    list = []
                    for i in range(6):  # ID
                        sample = torch.randn(1, latent_size).to(device)

                        for val in interpolation:
                            sample[0][z] = val
                            # print(sample)
                            list.append(sample.clone())
                    sample = torch.cat(list)
                    generate = model.decode(sample).cpu()

                    save_image(generate.view(args.latent_size * 6, chn_num, image_size, image_size),
                               'results/' + args.start_time + '/images/sample/' + str(epoch) + '_z' + str(
                                   z + 1) + '.png', nrow=args.latent_size)

    if args.fmnist:
        df_train = make_db('../data/FashionMNIST/train/*/*.png')
        df_test = make_db('../data/FashionMNIST/test/*/*.png')
    else:
        df_train = make_db('../data/noskin_all_v2/noskin_28/train_d/train/*.jpg')
        df_test = make_db('../data/noskin_all_v2/noskin_28/test_d/test/*.jpg')

    latant_space_exploration(df_train, 'train')
    latant_space_exploration(df_test, 'test')

    dict.update(model=model.to('cpu'))

    with open('./results/' + args.start_time + '/out.pickle', 'wb') as f:
        cloudpickle.dump(dict, f)
