from dataclasses import dataclass, asdict
import torch
import json
import os

def save_config(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Configuration saved to {config_path}")

@dataclass
class MemoryConfig:
    slots: int = 128
    dim: int = 128
    heads: int = 8
    topk: int = 16
    policy: str = "topk"  # "topk", "lru", "learned"
    decay_rate: float = 0.99
    use_decay_gate: bool = True
    bottleneck_dim: int = 64  # Dimension for semantic compression
    age_decay: float = 0.995  # Rate at which slot age increases

@dataclass
class ModelConfig:
    vocab_size: int = 20
    embed_dim: int = 128
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads_attn: int = 8
    max_seq_len: int = 512

@dataclass
class TaskConfig:
    seq_len: int = 10
    delay_len: int = 100

@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 50
    grad_clip: float = 1.0
    lambda_sparsity: float = 0.02
    lambda_diversity: float = 0.01
    lambda_forgetting: float = 0.005
    lambda_utilization: float = 0.01  # Weight for Slot Utilization Loss
    patience: int = 10
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