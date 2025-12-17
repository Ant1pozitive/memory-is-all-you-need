from torch.utils.data import Dataset
import torch
import numpy as np

class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, delay_len, size=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.delay_len = delay_len
        self.delim = vocab_size - 1
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq = np.random.randint(0, self.vocab_size - 1, size=self.seq_len)
        input_seq = np.concatenate([seq, np.zeros(self.delay_len), [self.delim], np.zeros(self.seq_len)])
        target = np.concatenate([np.full(self.seq_len + self.delay_len + 1, -100), seq])
        return torch.LongTensor(input_seq), torch.LongTensor(target)
