from dataclasses import dataclass
import torch

@dataclass
class MemoryConfig:
    slots: int = 64
    dim: int = 64
    heads: int = 4
    topk: int = 8
    policy: str = "topk"  # "topk", "lru", "learned"
    lru_capacity: int = 64  # for lru

@dataclass
class ModelConfig:
    vocab_size: int = 12
    embed_dim: int = 32
    hidden_dim: int = 128
    controller_type: str = "gru"  # "gru" or "transformer"

@dataclass
class TaskConfig:
    seq_len: int = 6
    delay_len: int = 40
    assoc_pairs: int = 5  # for associative recall

@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 30
    grad_clip: float = 1.0
    lambda_sparsity: float = 0.01
    lambda_diversity: float = 0.005
    lambda_usage: float = 0.001
    lambda_priority: float = 0.001  # for learned policy
    patience: int = 5  # early stopping
    use_wandb: bool = False
    mixed_precision: bool = True

@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    model: ModelConfig = ModelConfig()
    memory: MemoryConfig = MemoryConfig()
    task: TaskConfig = TaskConfig()
    train: TrainConfig = TrainConfig()

cfg = BaseConfig()
