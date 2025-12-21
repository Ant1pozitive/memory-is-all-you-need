from dataclasses import dataclass
import torch

@dataclass
class MemoryConfig:
    # Basic Structure
    slots: int = 128
    dim: int = 128
    heads: int = 8
    topk: int = 16
    
    # Policies
    # "meta" enables the dynamic mixing of Top-K, Uniform, and Random strategies
    policy: str = "meta"  
    
    # Decay & Age
    decay_rate: float = 0.99
    use_decay_gate: bool = True
    age_decay: float = 0.995
    
    # Semantic Compression
    bottleneck_dim: int = 64
    
    # Neural Synthesis (Dreaming)
    n_synthesis_layers: int = 2
    synthesis_heads: int = 4
    synthesis_interval: int = 4  # Run synthesis every N steps
    
    # Hebbian Graph Memory
    use_hebbian_graph: bool = True
    hebbian_lr: float = 0.05       # Learning rate for graph connections
    hebbian_decay: float = 0.995   # Decay factor for graph edges (forgetting old links)
    graph_influence: float = 0.2   # How much the graph affects the read operation (alpha)

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
    
    # Loss Weights
    lambda_sparsity: float = 0.02
    lambda_diversity: float = 0.01
    lambda_utilization: float = 0.01
    lambda_hallucination: float = 0.1  # Reconstruction loss weight
    
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