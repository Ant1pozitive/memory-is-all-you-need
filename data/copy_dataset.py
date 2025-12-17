from .base import TaskDataset
import numpy as np
import torch

class CopyDataset(TaskDataset):
    """Copy-with-delay dataset. Infinite generation."""
    def __init__(self, vocab_size, seq_len, delay_len):
        super().__init__(vocab_size)
        self.seq_len = seq_len
        self.delay_len = delay_len
        self.delim = vocab_size - 1

    def __next__(self):
        seq = np.random.randint(0, self.vocab_size - 1, size=(self.seq_len,))
        input_seq = np.concatenate([
            seq,
            np.zeros(self.delay_len, dtype=np.int64),
            np.array([self.delim], dtype=np.int64),
            np.zeros(self.seq_len, dtype=np.int64)
        ])
        target = np.concatenate([
            np.full(self.seq_len + self.delay_len + 1, -100, dtype=np.int64),
            seq
        ])
        return torch.from_numpy(input_seq).long(), torch.from_numpy(target).long()
