import matplotlib.pyplot as plt
import numpy as np

def loss_plot(filename):
    """
    Plots the loss values of the generator and discriminator through training.
    param:
        filename (str): file containing loss values. Formatting should correspond
        to the one defined in train.py.
    """
    data = np.loadtxt(filename, delimiter=', ', skiprows=1)
    epoch, batch, gen_loss, dis_loss = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, len(batch)-1, len(batch)), gen_loss, 'bo-', label="Generator loss")
    ax.plot(np.linspace(0, len(batch)-1, len(batch)), dis_loss, 'ro-', label="Discriminator loss")
    ax.set_title("Model losses evolution through training")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    _, epoch_count = np.unique(epoch, return_counts=True)
    for i in range(0, len(epoch_count)-1):
        ax.axvline(x=np.cumsum(epoch_count)[i], color='black', linestyle='--', label='epochs')
    ax.legend(['Generator loss', 'Discriminator loss', 'Epochs'])
    fig.show()

