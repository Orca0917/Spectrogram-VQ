import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MelDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [os.path.join(root_dir, f)
                      for f in os.listdir(root_dir)
                      if f.endswith(".npy")]
        assert len(self.files) > 0, f"No .npy files found in {root_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx])  # (n_mels, T)
        mel = torch.from_numpy(mel).float()
        mel = mel.unsqueeze(0)

        if mel.shape[2] % 2 == 1:
            mel = F.pad(mel, (0, 1), "constant", 0)

        return mel


def collate_mel(batch):
    lengths = torch.tensor([mel.shape[-1] for mel in batch], dtype=torch.long)
    T_max = max(lengths).item()

    padded_batch = []
    for mel in batch:
        pad_T = T_max - mel.shape[-1]
        if pad_T > 0:
            mel = torch.nn.functional.pad(mel, (0, pad_T))
        padded_batch.append(mel)

    padded_mels = torch.stack(padded_batch)
    return padded_mels


def load_data(args):
    dataset = MelDataset(args.dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_mel
    )
    return loader
