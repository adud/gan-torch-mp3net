"""
File containing all preprocessing functions useful
for preparing audio to go through the discriminator.
"""

import os
import librosa
import soundfile
from mdct import mdct
from window import window_vorbis
import numpy as np


def make_audio_chunks(file, chunk_duration,
                      path='./',  dest_path='./',
                      out_sr=None, make_mono=False):
    """
    Cut file.wav into several .wav files of duration chunk_duration.
    Audio is resampled to out_sr if need be.
    """
    data, rate = librosa.load(path+file, sr=out_sr, mono=make_mono)
    chunk_samples = int(chunk_duration*rate)
    data_duration = librosa.get_duration(data, sr=rate)
    data_samples = int(data_duration*rate)
    if make_mono:
        data = data[np.newaxis, :]
    if data_samples < chunk_samples:
        print(f"Audio file lasts less than {chunk_duration} seconds.")
        return 1
    for cnt in range(data_samples//chunk_samples):
        tmp = data[:, cnt*chunk_samples:(cnt+1)*chunk_samples]
        chunk_name = dest_path + file[:-4]+f"_{cnt}.wav"
        #chunk_name = chunk_name.replace(' ', '')
        #chunk_name = chunk_name.replace('-', '')
        #chunk_name = chunk_name.replace('_', '')
        # tmp needs to be transposed to have the shape expected by sf.write
        soundfile.write(chunk_name, tmp.T, samplerate=rate)
    return 0


def split_files(folder, duration, dest_path, rate, make_mono=False):
    """
    Split all .wav files from folder into .wav files of duration and rate.
    """
    for file in os.listdir(folder):
        print(f"Splitting {file}")
        make_audio_chunks(file, duration, path=folder,
                          dest_path=dest_path, out_sr=rate,
                          make_mono=make_mono)
    return 0


def apply_mdct(data, nfft=256, window=None):
    """
    Wrapper function calling mdct.mdct with the appropriate parameters.
    Defaults are correct values for 5 seconds files.
    """
    if window is None:
        window = window_vorbis(nfft)
    if len(data.shape) == 2:
        # if signal is stereo, must be transposed
        data = data.T
    out = mdct(data, framelength=nfft, window=window)
    return out
