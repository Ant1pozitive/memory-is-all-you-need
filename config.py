from dataclasses import dataclass


@dataclass
class BaseConfig:
device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


# model
vocab_size: int = 12
embed_dim: int = 32
hidden_dim: int = 128


# memory
mem_slots: int = 64
mem_dim: int = 64
mem_heads: int = 4
topk: int = 8


# data / tasks
seq_len: int = 6
delay_len: int = 40


# training
batch_size: int = 64
lr: float = 3e-4
epochs: int = 30
grad_clip: float = 1.0


cfg = BaseConfig()
