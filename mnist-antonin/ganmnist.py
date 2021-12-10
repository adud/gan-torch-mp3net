import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import numpy as np
import random

#random.seed(42)
#torch.manual_seed(42)

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 32

train_dataloader = DataLoader(
    training_data, batch_size=batch_size,
    num_workers=2
)

for X, y in train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# hyperparams

nz = 32  # dimensions of the latent space
nft = 2  # number of features in hidden layout


def weights_init(mod):
    if isinstance(mod, nn.BatchNorm2d):
        nn.init.normal_(mod.weight.data, 1.0, 0.02)
        nn.init.constant_(mod.bias.data, 0)
    elif isinstance(mod, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(mod.weight.data, 0.0, 0.02)
    elif isinstance(mod, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid,
                          nn.Sequential, Generator, Discriminator)):
        pass
    else:
        print(f"unknown {type(mod)}")


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, nft * 2, 7, bias=False),
            nn.BatchNorm2d(nft * 2),
            nn.ReLU(True),
            # output size (nft * 4) x 7 x 7
            nn.ConvTranspose2d(nft * 2, nft, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nft),
            nn.ReLU(True),
            # output size (nft * 2) x 14 x 14
            nn.ConvTranspose2d(nft, 1, 4, 2, 1, bias=False),
            # output size 1 x 28 x 28
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


sl = 1e-3  # LeakyReLU Slope


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, nft, 4, 2, 1, bias=False),
            nn.LeakyReLU(sl, nft),

            nn.Conv2d(nft, nft * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nft * 2),
            nn.LeakyReLU(sl, nft * 2),

            nn.Conv2d(nft * 2, 1, 7, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


gen = Generator()
gen.apply(weights_init)

dis = Discriminator()
dis.apply(weights_init)


lossFun = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1)

real_label = 1
gen_label = 0


lr = 2e-4
beta1 = 0.5

optDis = optim.Adam(dis.parameters(), lr=lr, betas=(beta1, 0.999))
optGen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))


epochs = 10

trainLabels = torch.full((batch_size, ), real_label, dtype=torch.float)

for epoch in range(epochs):
    print(f"#### epoch: {epoch} ####")
    for i, dat in enumerate(train_dataloader):
        # # optimize discriminator
        dis.zero_grad()

        pict = dat[0]
        # GANHacks Idea: separate training
        
        # all real

        out = dis(pict).view(-1)
        resDisR = out.mean().item()

        errDisReal = lossFun(out, trainLabels)
        errDisReal.backward()

        # all fake
        latVec = torch.randn(batch_size, nz, 1, 1)
        genDat = gen(latVec).detach()

        trainLabels.fill_(gen_label)
        out = dis(genDat).view(-1)
        resDisG = out.mean().item()

        errDisGen = lossFun(out, trainLabels)
        errDisGen.backward()
        errDis = errDisGen + errDisReal
        optDis.step()

        # # optimize generator

        for j in range(20):
            gen.zero_grad()
            # generator wins if discriminator thinks its output are real
            trainLabels.fill_(real_label)
            out = dis(genDat).view(-1)
            errGen = lossFun(out, trainLabels)
            resGen = out.mean().item()
            errGen.backward()
            optGen.step()

        if not(i % 50):
            print(f"errDis: {errDis.item()}\nerrGen: {errGen.item()}")
            with torch.no_grad():
                fake = gen(fixed_noise).detach()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                plt.imsave("test.png", np.transpose(np.array(grid), (1, 2, 0)))
