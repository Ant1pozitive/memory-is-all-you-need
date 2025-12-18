from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def topk_sparse_softmax(sim: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Efficient Top-K sparse softmax for memory addressing"""
    B, H, N = sim.shape
    sim_flat = sim.view(B * H, N)
    _, idx = torch.topk(sim_flat, k=min(k, N), dim=-1)
    mask = torch.zeros_like(sim_flat, dtype=torch.bool)
    arange = torch.arange(B * H, device=sim.device).unsqueeze(1)
    mask[arange, idx] = True
    mask = mask.view(B, H, N)
    masked = sim.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked, dim=-1)
    return weights, mask

class MultiHeadMemoryBank(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, n_heads: int = 8, topk: int = 16,
                 policy: str = "topk", use_decay_gate: bool = True, decay_rate: float = 0.99,
                 bottleneck_dim: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.topk = topk
        self.policy = policy
        self.use_decay_gate = use_decay_gate
        self.decay_rate = decay_rate

        # Information Bottleneck: Compresses information before writing to memory
        self.bottleneck = nn.Sequential(
            nn.Linear(slot_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, slot_dim)
        )

        self.init_memory = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.01)
        nn.init.orthogonal_(self.init_memory)

        # Buffers for Age-based forgetting and utilization tracking
        self.register_buffer("age", torch.zeros(1, num_slots))
        
        # LayerNorm for reading stability
        self.norm = nn.LayerNorm(slot_dim)
        self.head_merge = nn.Linear(n_heads * slot_dim, slot_dim)

        if use_decay_gate:
            self.decay_gate = nn.Parameter(torch.ones(num_slots) * 0.99)

        if policy == "learned":
            self.priority_mlp = nn.Linear(slot_dim, 1)

    def reset_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        self.age = torch.zeros(batch_size, self.num_slots, device=device)
        return self.init_memory.expand(batch_size, -1, -1).clone().to(device)

    @staticmethod
    def cosine_sim(keys: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        k_norm = F.normalize(keys, dim=-1)
        m_norm = F.normalize(memory, dim=-1)
        return torch.einsum('bhd,bnd->bhn', k_norm, m_norm)

    def apply_decay(self, memory: torch.Tensor) -> torch.Tensor:
        if self.use_decay_gate:
            decay = torch.sigmoid(self.decay_gate).view(1, -1, 1)
            memory = memory * decay
        else:
            memory = memory * self.decay_rate
        return memory

    def read(self, memory: torch.Tensor, read_keys: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = self.cosine_sim(read_keys, memory) * beta.unsqueeze(-1)
        weights, mask = topk_sparse_softmax(sim, self.topk)

        read_per_head = torch.einsum('bhn,bnd->bhd', weights, memory)
        read_combined = self.head_merge(read_per_head.reshape(read_per_head.shape[0], -1))
        
        # Apply LayerNorm for numerical stability across deep timesteps
        read_combined = self.norm(read_combined)

        # Update Age: slots that are read have their age reset
        with torch.no_grad():
            accessed_slots = mask.any(dim=1).float() # [B, N]
            self.age = self.age * (1 - accessed_slots) + accessed_slots * 0.0

        return read_combined, weights

    def write(self, memory: torch.Tensor, write_keys: torch.Tensor, write_vals: torch.Tensor,
              erase: torch.Tensor, add_gate: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Age-based logic: Increment age for all slots
        with torch.no_grad():
            self.age += 1.0

        # Apply Bottleneck to write values
        compressed_vals = self.bottleneck(write_vals)

        sim = self.cosine_sim(write_keys, memory) * beta.unsqueeze(-1)

        # If policy is LRU, bias writing towards older slots
        if self.policy == "lru":
            age_bias = (self.age / (self.age.max() + 1e-8)).unsqueeze(1)
            sim = sim + age_bias

        if self.policy == "learned":
            priorities = self.priority_mlp(memory).squeeze(-1).unsqueeze(1)
            sim = sim + priorities

        weights, _ = topk_sparse_softmax(sim, self.topk)

        w_unsq = weights.unsqueeze(-1)
        erase_unsq = erase.unsqueeze(-1).unsqueeze(-1)
        add_unsq = add_gate.unsqueeze(-1).unsqueeze(-1) * compressed_vals.unsqueeze(2)

        mem_exp = memory.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mem_after_erase = mem_exp * (1 - w_unsq * erase_unsq)
        
        # New memory is a mean of head updates
        new_memory = mem_after_erase + w_unsq * add_unsq
        new_memory = new_memory.mean(dim=1)
        
        # Final normalization and decay
        new_memory = F.normalize(new_memory + 1e-8, dim=-1)
        new_memory = self.apply_decay(new_memory)

        return new_memory, weights