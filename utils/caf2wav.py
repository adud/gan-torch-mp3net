"""
Helping functions for converting .caf audio files to .wav
Currently Linux only.
"""

import os


def caf2wav(input_name, output_name, options=''):
    """
    Function calling the correct OS command for converting
    input_name.caf to output_name.wav.
    options are for ffmpeg if some arguments need to be added.
    Linux only as of this version (uses ffmpeg).
    """
    linux = True
    # automatically checking the OS should be added as an improvement later
    if linux:
        cmd = f"ffmpeg -i '{input_name}' {options} '{output_name}'"
        # Error catching should be added
        os.system(cmd)
    return 0


def convert_batch(source_folder='./', destination_folder='./', options=''):
    """
    Function to convert all .caf files from source_folder to .wav files
    in destination_folder. options can be passed to caf2wav if needed.
    """
    for file in os.listdir(source_folder):
        if file.endswith(".caf") or file.endswith(".CAF"):
            out_path = destination_folder + file[:-4] + ".wav"
            in_path = source_folder + file
            caf2wav(in_path, out_path, options)
    return 0
