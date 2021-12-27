"""
Dataloader definition according to PyTorch recommendations:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset

class MP3NetDataset(Dataset):
    def __init__(self, snd_dir, transform=None, target_transform=None):
        self.snd_dir = snd_dir
        self.transform = transform
        self.target_transform = target_transform
        # might be computationnally expensive, should we use
        # an external .csv file?
        self.file_names = [f for f in os.listdir(snd_dir)
                           if f.endswith('.wav')
                           and os.path.isfile(os.path.join(snd_dir, f))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        snd_path = os.path.join(self.snd_dir, self.file_names[idx])
        sound, _ = torchaudio.load(snd_path)
        if self.transform:
            sound = self.transform(sound)
        return sound
