from .base import TaskDataset
import numpy as np
import torch

class AssociativeRecallDataset(TaskDataset):
    """Associative recall: pairs (key-value), then query key, expect value."""
    def __init__(self, vocab_size, num_pairs, delay_len, size=None):
        super().__init__(vocab_size, size)
        self.num_pairs = num_pairs
        self.delay_len = delay_len
        self.delim = vocab_size - 1
        self.query_marker = vocab_size - 2

    def __getitem__(self, idx):
        keys = np.random.randint(0, self.vocab_size - 2, size=(self.num_pairs,))
        values = np.random.randint(0, self.vocab_size - 2, size=(self.num_pairs,))
        query_idx = np.random.randint(0, self.num_pairs)
        input_seq = np.concatenate([
            np.ravel(np.stack((keys, values), axis=1)),  # flatten pairs
            np.zeros(self.delay_len, dtype=np.int64),
            np.array([self.query_marker], dtype=np.int64),
            np.array([keys[query_idx]], dtype=np.int64),
            np.zeros(self.num_pairs, dtype=np.int64)  # output space
        ])
        target = np.concatenate([
            np.full(len(input_seq) - self.num_pairs, -100, dtype=np.int64),
            values
        ])
        return torch.from_numpy(input_seq).long(), torch.from_numpy(target).long()
