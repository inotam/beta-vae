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

now = datetime.datetime.now()
start_time =  now.strftime('%Y%m%d-%H:%M:%S')

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
parser.add_argument('--start-time', type=str, default=start_time, metavar='ST', help='(default: today_time)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

with open('../data/train.pickle','rb') as f:
    dataset_train = pickle.load(f)

with open('../data/test.pickle', 'rb') as f:
    dataset_test = pickle.load(f)

os.mkdir("./results/"+args.start_time+'/')
os.mkdir("./results/"+args.start_time+'/images/')
os.mkdir("./results/"+args.start_time+'/images/train')
os.mkdir("./results/"+args.start_time+'/images/test')
os.mkdir("./results/"+args.start_time+'/images/sample')

dict = {'hyper-parameter':args}

train_loader = torch.utils.data.DataLoader(
    #datasets.FashionMNIST('../data', train=True, download=True,transform=transforms.ToTensor()),
    dataset_train,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    #datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    dataset_test,
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        latent = args.latent_size
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent)
        self.fc22 = nn.Linear(400, latent)
        self.fc3 = nn.Linear(latent, 400)
        self.fc4 = nn.Linear(400, 784)

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

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

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

    if epoch % 10 == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        print('====> Epoch: {} Average BCE: {:.4f}'.format(
            epoch, train_bce / len(train_loader.dataset)))
        print('====> Epoch: {} Average KLD: {:.4f}'.format(
              epoch, train_kld / len(train_loader.dataset)))

    if epoch == args.epochs:
        dict.update(train_loss = train_loss,train_bce=train_bce, train_kld = train_kld)



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

    if epoch == args.epochs:
        dict.update(test_loss = test_loss, test_bce =test_bce, test_kld = test_kld)

if __name__ == "__main__":
    latent_size = args.latent_size
    interpolation = torch.arange(-3, 3 + 0.1, 2 / 3)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:
            with torch.no_grad():
                for z in range(10):
                    list=[]
                    for i in range(6): #ID
                        sample = torch.randn(1, latent_size).to(device)

                        for val in interpolation:
                            sample[0][z] = val
                            #print(sample)
                            list.append(sample.clone())
                    sample = torch.cat(list)
                    generate = model.decode(sample).cpu()

                    save_image(generate.view(60, 1, 28, 28),
                        'results/' + args.start-start_time + '/images/sample/' + str(epoch) + '_z'+ str(z+1)+'.png',nrow=10)

    dict.update(model=model.to('cpu'))


