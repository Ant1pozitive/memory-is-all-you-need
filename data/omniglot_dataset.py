from torch.utils.data import Dataset
import numpy as np
import torch
from .base import TaskDataset

class OmniglotDataset(TaskDataset):
    """Simulated Omniglot few-shot continual classification."""
    def __init__(self, cfg, split: str = 'train'):
        super().__init__(cfg.model.vocab_size)
        self.num_classes = cfg.task.num_classes
        self.shots = 5
        self.query_len = 3
        self.size = 5000 if split == 'train' else 500

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        labels = np.arange(self.num_classes)
        np.random.shuffle(labels)
        support = np.repeat(labels, self.shots)
        query_labels = np.random.choice(labels, self.query_len)
        input_seq = np.concatenate([support, [self.delim], query_labels * 0])
        target = np.concatenate([np.full(len(support) + 1, -100), query_labels])
        return torch.from_numpy(input_seq).long(), torch.from_numpy(target).long()
