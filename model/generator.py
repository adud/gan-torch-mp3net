import numpy as np
import torch
from torch import nn



from torch.nn.modules.activation import LeakyReLU, ReLU
from torch.nn.modules.conv import ConvTranspose2d

class Generator_block(nn.Module):
    def __init__(self, channels_in, channels_mid, channels_out):
        super(Generator_block, self).__init__()
        self.channels_in = channels_in
        self.channels_mid = channels_mid
        self.channels_out = channels_out
        
        self.left_block =  nn.Sequential(
            nn.ConvTranspose2d(self.channels_in, self.channels_mid, (2,1), (2,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels_mid, self.channels_mid, 1, (1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels_mid, self.channels_mid, 1, (1,1))   
        )
        self.right_block = nn.Sequential(
            nn.ConvTranspose2d(self.channels_in, self.channels_mid, 2, (2,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels_mid, self.channels_mid, 1, (1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels_mid, self.channels_mid, 1, (1,1))  
        )
        
        self.concatenated_block = nn.Sequential(
            nn.ConvTranspose2d(self.channels_mid, self.channels_out, (2,1), (2,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels_out , self.channels_out , 1 , (1,1)),
            nn.LeakyReLU()
        )

    def forward(self, input):
        input_high = torch.clone(input[:,:,:,int(input.size(3)/2):])        
        input_concatenated = torch.cat((self.left_block(input),self.right_block(input_high)),3)  

        return self.concatenated_block(input_concatenated)

class Generator(nn.Module):
    def __init__(self, out_channels):
        super(Generator, self).__init__()
        self.out_channels = out_channels
        self.main = nn.Sequential(
            Generator_block(512, 512, 512),
            Generator_block(512, 512, 256),
            Generator_block(256, 256, 128),
            Generator_block(128, 128, 64),
            Generator_block(64, 32, 16),
            nn.Conv2d(16, self.out_channels, (1, 1)),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
