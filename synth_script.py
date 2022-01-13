import torch
import mdct
import soundfile
import numpy as np

inp = torch.load('e/b.zip')
mdc = np.array(inp[0,0,:,:])
y = mdct.imdct(mdc.T, framelength=256, hopsize=None, overlap=2, centered=True, padding=0)
