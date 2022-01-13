import mdct
import torch
import soundfile
import os, os.path
import numpy as np
from model.generator import Generator
from analysis import loss_plot, mdct_plot

gen = Generator(1)

svg_path = input("Where are the model parameters saved? (relative path)")
svg_path = svg_path + '/model_parameters/'

num_files = len([name for name in os.listdir(svg_path) if os.path.isfile(name)])//2
epoch = input(f"Which epoch? (0-{num_files})")

#  Load model parameters
state_dict = torch.load(svg_path + f'generator_{epoch}')
gen.load_state_dict(state_dict)

#  Generate outputs

out = int(input("How many outputs do you want to generate?"))
if out <= 0:
    exit()

while True:
    ans = input("Do you want to plot the mdct spectrogram? [Y/n]")
    if ans.upper() in ['', 'Y']:
        mdct_flag = True
        break
    elif ans.upper() == 'N':
        mdct_flag = False
    else:
        print("Please enter 'y' or 'n'")
        continue

while True:
    ans = input("Do you want to generate and save .wav files? [Y/n]")
    if ans.upper() in ['', 'Y']:
        wav_flag = True
        out_path = input("Where to?")
        if not(os.path.exists(out_path)):
            os.makedirs(out_path)
        break
    elif ans.upper() == 'N':
        wav_flag = False
        break
    else:
        print("Please enter 'y' or 'n'")
        continue

for i in range(out):
    noise = torch.randn(1, 512, 1, 4)
    with torch.no_grad():
        gen_mdct = gen(noise)
    gen_mdct = gen_mdct.detach().numpy()
    if mdct_flag:
        mdct_plot.mdct_plot(gen_mdct[0,0,:,:], num=i)
    if wav_flag:
        audio = mdct.imdct(gen_mdct[0,0,:,:].T, framelength=256, hopsize=None, overlap=2, centered=True, padding=0)
        wav_name = f"gen_{i}.wav"
        soundfile.write(out_path + '/' + wav_name, audio, 22050)
        print(wav_name, " saved.")

ans = input("All done!")
exit()






