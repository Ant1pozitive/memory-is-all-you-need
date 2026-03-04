from torch.utils.data import Dataset
import numpy as np
import torch
from .base import TaskDataset

class AssociativeRecallDataset(TaskDataset):
    """Classic Associative Recall task for multi-task continual learning."""
    def __init__(self, cfg, split: str = 'train'):
        super().__init__(cfg.model.vocab_size)
        self.num_pairs = cfg.task.num_pairs
        self.delay_len = cfg.task.delay_len
        self.delim = cfg.model.vocab_size - 1
        self.query_marker = cfg.model.vocab_size - 2
        self.size = 10000 if split == 'train' else 1000

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        keys = np.random.randint(0, self.vocab_size - 2, size=(self.num_pairs,))
        values = np.random.randint(0, self.vocab_size - 2, size=(self.num_pairs,))
        query_idx = np.random.randint(0, self.num_pairs)
        input_seq = np.concatenate([
            np.ravel(np.stack((keys, values), axis=1)),
            np.zeros(self.delay_len, dtype=np.int64),
            np.array([self.query_marker]),
            np.array([keys[query_idx]]),
            np.zeros(self.num_pairs, dtype=np.int64)
        ])
        target = np.concatenate([
            np.full(len(input_seq) - self.num_pairs, -100, dtype=np.int64),
            values
        ])
        return torch.from_numpy(input_seq).long(), torch.from_numpy(target).long()
