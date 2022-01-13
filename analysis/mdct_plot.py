import matplotlib.pyplot as plt
import numpy as np

def mdct_plot(mdct, num=0):
    """
    Plots mdct with correct arguments for imshow to have figures similar
    to the mp3net paper.
    param:
        mdct (np.arr): mdct to plot
    """
    fig, ax = plt.subplots(num=num)
    spectrogram = np.log(np.power(mdct/np.max(np.abs(mdct)), 2) + 1e-10)
    ax.imshow(spectrogram, cmap='plasma', aspect='auto', 
              interpolation=None, origin="lower")
    fig.show()



