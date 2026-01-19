from dataclasses import dataclass, field
import torch

@dataclass
class MemoryConfig:
    # Basic Structure
    slots: int = 128      # Total slots if not using split (legacy)
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
    hebbian_decay: float = 0.995   # Decay factor for graph edges
    graph_influence: float = 0.2   # Alpha

    # STM/LTM Separation
    stm_slots: int = 64
    ltm_slots: int = 1024
    stm_decay_rate: float = 0.95   # Faster decay for STM (volatile)
    ltm_decay_rate: float = 0.995  # Slower decay for LTM (stable)
    consolidation_threshold: float = 0.1  # Transfer if utilization > threshold
    
    # Explicit Forgetting
    prune_threshold: float = 0.1   # Edges below this are pruned
    forget_rate_multiplier: float = 2.0  # Multiplier for active forget decay

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
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

cfg = BaseConfig()