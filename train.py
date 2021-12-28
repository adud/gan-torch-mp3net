import torch
import torchaudio
from torch.utils.data import DataLoader
from config import *
import utils.dataset
import utils.transform

transforms = torch.nn.Sequential(
                        torchaudio.transforms.Fade(FADE_IN, FADE_OUT),
                        utils.transform.TransposeTransform(),
                        utils.transform.ZeroPadTransform(FFT_SIZE),
                        utils.transform.MdctTransform(FFT_SIZE),
                        utils.transform.ToTensorTransform()
                        )

training_data = utils.dataset.MP3NetDataset(TRAIN_PATH, transform=transforms)

train_dataloader = DataLoader(training_data, BATCH_SIZE, shuffle=True)
