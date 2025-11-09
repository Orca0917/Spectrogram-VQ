import os
import json
import torch
from scipy.io.wavfile import write
from models import hifigan


def load_hifigan_model(config_path, ckpt_path, device):
    with open(config_path, "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(ckpt_path)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    # vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def synthesize(vocoder, mels, save_path=None, sr=22050):
    wav = vocoder(mels).squeeze()
    wav = wav.cpu().detach().numpy()
    if save_path is not None:
        write(save_path, sr, wav)
    return wav