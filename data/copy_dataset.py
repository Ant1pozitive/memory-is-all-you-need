import torch
import numpy as np
from torch.utils.data import Dataset

class CopyDataset(Dataset):
    """
    Copy task dataset with delay.
    Supports train/val splits with different sizes and reproducible val.
    """
    def __init__(self, cfg, split: str = 'train'):
        self.cfg = cfg
        self.vocab_size = cfg.model.vocab_size
        self.seq_len = cfg.task.seq_len
        self.delay_len = cfg.task.delay_len
        self.delim = self.vocab_size - 1  # delimiter token

        if split == 'train':
            self.size = 10000  # large for training
        elif split == 'val':
            self.size = 1000   # smaller for validation
        else:
            raise ValueError(f"Unknown split: {split}")

        self.split = split

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Make validation reproducible
        if self.split != 'train':
            np.random.seed(42 + idx)

        seq = np.random.randint(0, self.vocab_size - 1, size=self.seq_len)
        zeros_delay = np.zeros(self.delay_len, dtype=np.int64)
        zeros_out = np.zeros(self.seq_len, dtype=np.int64)

        input_seq = np.concatenate([seq, zeros_delay, [self.delim], zeros_out])
        target = np.concatenate([
            np.full(self.seq_len + self.delay_len + 1, -100, dtype=np.int64),
            seq
        ])

        return torch.LongTensor(input_seq), torch.LongTensor(target)
