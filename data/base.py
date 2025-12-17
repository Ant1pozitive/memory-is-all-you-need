from torch.utils.data import IterableDataset
import numpy as np
import torch

class TaskDataset(IterableDataset):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __iter__(self):
        return self
