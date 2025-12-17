import numpy as np
import torch
from torch.utils.data import Dataset

class CopyDataset(Dataset):
"""Copy-with-delay dataset.
Timeline: [seq (L), zeros (delay), delim, zeros(L)]
Targets: ignore tokens until output phase then expect original seq
"""
def __init__(self, vocab_size, seq_len, delay_len, size=2000):
self.vocab_size = vocab_size
self.seq_len = seq_len
self.delay_len = delay_len
self.size = size
self.delim = vocab_size - 1

def __len__(self):
return self.size

def __getitem__(self, idx):
seq = np.random.randint(0, self.vocab_size - 1, size=(self.seq_len,), dtype=np.int64)
input_seq = np.concatenate([
seq,
np.zeros(self.delay_len, dtype=np.int64),
np.array([self.delim], dtype=np.int64),
np.zeros(self.seq_len, dtype=np.int64)
])
# targets: -100 to ignore before output phase
target = np.concatenate([
np.full(self.seq_len + self.delay_len + 1, -100, dtype=np.int64),
seq
])
return torch.from_numpy(input_seq).long(), torch.from_numpy(target).long()
