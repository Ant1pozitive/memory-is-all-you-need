from .base import TaskDataset
import numpy as np
import torch
# Note: For simplicity, simulate Omniglot as few-shot classification toy (no real images)

class OmniglotDataset(TaskDataset):
    """Simulated few-shot Omniglot: Present K classes, N shots, query."""
    def __init__(self, vocab_size, num_classes=5, shots=1, query_len=1):
        super().__init__(vocab_size)
        self.num_classes = num_classes
        self.shots = shots
        self.query_len = query_len
        self.delim = vocab_size - 1

    def __next__(self):
        labels = np.arange(self.num_classes)
        np.random.shuffle(labels)
        support = np.repeat(labels, self.shots)
        query_labels = np.random.choice(labels, self.query_len)
        input_seq = np.concatenate([support, [self.delim], query_labels * 0])  # placeholders for features
        target = np.concatenate([np.full(len(support) + 1, -100), query_labels])
        return torch.from_numpy(input_seq).long(), torch.from_numpy(target).long()
