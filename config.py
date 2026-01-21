from dataclasses import dataclass, field
import torch

@dataclass
class MemoryConfig:
    # Basic Structure
    slots: int = 128
    dim: int = 128
    heads: int = 8
    topk: int = 16
    
    # Policies
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
    synthesis_interval: int = 4
    
    # Hebbian Graph Memory & Topological Dynamics
    use_hebbian_graph: bool = True
    hebbian_lr: float = 0.05
    hebbian_decay: float = 0.995
    
    # Topological Memory
    # Number of hops for activation spreading (A -> B -> C)
    spreading_steps: int = 2  
    # How much the graph structure influences retrieval vs raw content
    graph_influence: float = 0.3 
    
    # STM/LTM Separation
    stm_slots: int = 64
    ltm_slots: int = 1024
    stm_decay_rate: float = 0.95
    ltm_decay_rate: float = 0.995
    
    # Curriculum Consolidation
    use_dynamic_consolidation: bool = True
    consolidation_threshold: float = 0.1
    consolidation_percentile: float = 0.85
    
    # Explicit Forgetting
    prune_threshold: float = 0.1
    forget_rate_multiplier: float = 2.0
    
    # Context-Dependent Distance Learning
    # If True, computes dynamic feature weights based on query context
    use_context_metric: bool = True

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
    lambda_hallucination: float = 0.1
    
    # Graph Regularization: Penalizes dense graphs to encourage sparse topology
    lambda_graph_sparsity: float = 0.005
    
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