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
    def __init__(self):
        super(Generator, self).__init__() 
        
        self.main = nn.Sequential(
            Generator_block(512,512,512),
            Generator_block(512,256,256),
            Generator_block(256,128,128),
            nn.Conv2d(128, 2, (1,1)),
            nn.Tanh()
        )
    
    def forward(self,input):
        return self.main(input)
    
class Generator_old(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
    def forward_block(self, input , channels_mid, channels_out):
        channels_in = input.shape[1]
        
        left_block =  nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_mid, (2,1), (2,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels_mid, channels_mid, 1, (1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels_mid, channels_mid, 1, (1,1))   
        )
        right_block = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_mid, 2, (2,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels_mid, channels_mid, 1, (1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels_mid, channels_mid, 1, (1,1))  
        )
        input_high = torch.clone(input[:,:,:,int(input.size(3)/2):])
        print(left_block(input).shape)
        print(right_block(input_high).shape)
        
        input_concatenated = torch.cat((left_block(input),right_block(input_high)),3)
        
        concatenated_block = nn.Sequential(
            nn.ConvTranspose2d(channels_mid, channels_out, (2,1), (2,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels_out , channels_out , 1 , (1,1)),
            nn.LeakyReLU()
        )
        return concatenated_block(input_concatenated)  
    
    def forward(self,input):
        stage0 = self.forward_block(input, 512 , 512)
        print(stage0.shape)
        stage1 = self.forward_block(stage0, 256, 256)
        #stage2 = self.forward_block(stage1, 128, 128)
        #stage3 = self.forward_block(stage2, 128, 64)  
        #stage4 = self.forward_block(stage3, 32, 16)
        #stage5 = self.forward_block(stage4, 8, 4)
        
        end_block = nn.Sequential(
            nn.Conv2d(256, 2, (1,1)),
            nn.Sigmoid()
        )
        return end_block(stage1)
    
    def forward1(self,input):
        #output = self.forward_block(input, 512 , 512)
        #output = self.forward_block(output, 256, 256)
        #output = self.forward_block(output, 128, 128)
        #output = self.forward_block(output, 128, 64)  
        #output = self.forward_block(output, 32, 16)
        #output = self.forward_block(output, 8, 4)
        
        end_block = nn.Sequential(
            self.forward_block(512 , 512),
            self.forward_block(256, 256),
            nn.Conv2d(256, 2, (1,1)),
            nn.Sigmoid()
        )
        return end_block(input)
