import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_spectrogram_to_img(mels, filepath, nrow=4, normalize=True, cmap="viridis"):
    if isinstance(mels, torch.Tensor):
        mels = mels.detach().cpu().numpy()

    B = mels.shape[0]

    ncol = nrow
    nrow = int(np.ceil(B / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 2.5))
    axes = np.array(axes).reshape(nrow, ncol)

    for idx, ax in enumerate(axes.flatten()):
        if idx >= B:
            ax.axis("off")
            continue
        
        mel = mels[idx, 0, :, :]
        if normalize:
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        ax.imshow(mel, origin="lower", aspect="auto", cmap=cmap)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)