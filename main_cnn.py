from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
import pickle

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

with open('../data/train.pickle', 'rb') as f:
    dataset_train = pickle.load(f)

with open('../data/test.pickle', 'rb') as f:
    dataset_test = pickle.load(f)

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
    #dataset_train,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    #dataset_test,
    batch_size=args.batch_size, shuffle=True, **kwargs)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAE(nn.Module):
    def __init__(self, z_dim=10, nc= 1):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.nc = nc
        self.fc21 = nn.Linear(256, self.z_dim)
        self.fc22 = nn.Linear(256, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 256)


        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64, 7, 7
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 1),  # B,  64,  4,  4
            #n.Linear(256, z_dim * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 1),  # B,  32, 7, 7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 28, 28
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        #h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        #mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        #print(z)
        return self.decode(z), mu, logvar


model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + args.beta*KLD, BCE, KLD


def train(epoch):
    model.train()
    train_loss = 0
    train_bce = 0
    train_kld = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
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
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        print('====> Epoch: {} Average BCE: {:.4f}'.format(
            epoch, train_bce / len(train_loader.dataset)))
        print('====> Epoch: {} Average KLD: {:.4f}'.format(
              epoch, train_kld / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    test_bce = 0
    test_kld = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss
            test_bce += bce
            test_kld += kld
            if i == 0 and (epoch % 10 ==0):
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    #test_loss /= len(test_loader.dataset)
    if epoch % 10 ==0:
        print('====> Test set loss: {:.4f}'.format(test_loss/len(test_loader.dataset)))
        print('====> Test set BCE: {:.4f}'.format(test_bce/ len(test_loader.dataset)))
        print('====> Test set KLD: {:.4f}'.format(test_kld / len(test_loader.dataset)))


if __name__ == "__main__":
    latent_size = args.latent_size
    interpolation = torch.arange(-2, 2 + 0.1, 4 / 9)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:
            with torch.no_grad():
                for z in range(10):
                    list=[]
                    for i in range(6): #ID
                        sample = torch.randn(1, latent_size).to(device)
                        list_tensor = []
                        for val in interpolation:
                            sample[0][z] = val
                            list_tensor.append(sample)
                        sample = torch.cat(list_tensor)
                        sample = model.decode(sample).cpu()
                        list.append(sample)
                    #print(list.size())
                    sample = torch.cat(list)

                    save_image(sample.view(60, 1, 28, 28),
                        'results/sample_' + str(epoch) + '_z'+ str(z+1)+'.png',nrow=10)
