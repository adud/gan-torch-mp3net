import torch
from torch import nn, optim

class Discriminator_block(nn.Module):
    """
    Class defining the discriminator for mp3net.
    """
    def __init__(self, channels, std=False):
        super().__init__()
        self.channels = channels
        self.std = std
        self.start_block = nn.Sequential(
                    nn.Conv2d(self.channels[0], self.channels[0], kernel_size=1, bias=False),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(self.channels[0], self.channels[1], (2, 1), stride=(2, 1), bias=False),
                    nn.LeakyReLU(inplace=True),
                    )

        self.left_block = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[1], 1, bias=False),
            nn.Conv2d(self.channels[1], self.channels[1], 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.channels[1], self.channels[2], (2, 1), stride=(2,1), bias=False),
            nn.LeakyReLU(inplace=True)
        )

        self.right_block = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[1], 1, bias=False),
            nn.Conv2d(self.channels[1], self.channels[1], 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.channels[1], self.channels[2], 2, stride=2, bias=False)
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        tmp = self.start_block(inputs)
        tmp_left = torch.clone(tmp[:, :, :, 0:int(tmp.shape[3]/2)])
        tmp_right = torch.clone(tmp[:, :, :, int(tmp.shape[3]/2):])
        left = self.left_block(tmp_left)
        right = self.right_block(tmp_right)
        end = torch.clone(left)
        end[:, :, :, int(end.shape[3]/2):] += right
        end[:, :, :, int(end.shape[3]/2):] /= 2
        if self.std:
            end = torch.cat((end, torch.std(end, 1, True).view(batch_size, 1, end.shape[2], -1)), dim=1)
        return end

class Discriminator(nn.Module):
    def __init__(self, in_channels, h_dim):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = h_dim
        self.model = nn.Sequential(
                        nn.Conv2d(self.in_channels, self.hidden_dim, 1, bias=False),
                        Discriminator_block((self.hidden_dim, 2*self.hidden_dim, 4*self.hidden_dim)),
                        Discriminator_block((4 * self.hidden_dim, 4*2 * self.hidden_dim, 4*4 * self.hidden_dim)),
                        Discriminator_block((4*4 * self.hidden_dim, 4*4*2 * self.hidden_dim, 4*4*2 * self.hidden_dim), std=True),
                        Discriminator_block((4*4*2 * self.hidden_dim + 1, 4*4*2 * self.hidden_dim, 4*4*2 * self.hidden_dim)),
                        nn.Flatten(),
                        nn.Linear(2*4 * 4*4*2 * self.hidden_dim, 1, bias=False),
                        nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)

    def filtering(self, inputs):
        outputs = inputs  # mettre les filtres psychoacoustiques
        return outputs



