from torch.utils.data import Dataset
import numpy as np
import torch

class TaskDataset(Dataset):
    def __init__(self, vocab_size, size=None):  # size=None for infinite
        self.vocab_size = vocab_size
        self.size = size or int(1e9)  # infinite

    def __len__(self):
        return self.size
