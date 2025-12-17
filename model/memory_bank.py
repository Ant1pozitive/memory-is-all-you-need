from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn

def topk_sparse_softmax(sim: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N = sim.shape
    sim_flat = sim.view(B * H, N)
    vals, idx = torch.topk(sim_flat, k=min(k, N), dim=-1)
    mask = torch.zeros_like(sim_flat, dtype=torch.bool)
    arange = torch.arange(B * H, device=sim.device).unsqueeze(1)
    mask[arange, idx] = True
    mask = mask.view(B, H, N)
    masked = sim.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked, dim=-1)
    return weights, mask

class MultiHeadMemoryBank(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, n_heads: int = 4, topk: int = 8, policy: str = "topk"):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.topk = topk
        self.policy = policy
        self.init_memory = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.01)
        nn.init.orthogonal_(self.init_memory)
        self.head_merge = nn.Linear(n_heads * slot_dim, slot_dim)  # for read aggregation
        if policy == "learned":
            self.priority_mlp = nn.Linear(slot_dim, 1)
        if policy == "lru":
            self.access_times = None

    def reset_memory(self, batch_size: int, device):
        mem = self.init_memory.expand(batch_size, -1, -1).contiguous().to(device)
        if self.policy == "lru":
            self.access_times = torch.zeros(batch_size, self.num_slots, dtype=torch.float32, device=device)
        return mem

    @staticmethod
    def cosine_sim(keys: torch.Tensor, memory: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        k_norm = keys.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        m_norm = memory.norm(p=2, dim=-1).clamp_min(eps)
        dot = torch.einsum('bhd,bnd->bhn', keys, memory)
        sim = dot / (k_norm * m_norm.unsqueeze(1) + eps)
        return sim

    def read(self, memory: torch.Tensor, read_keys: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = self.cosine_sim(read_keys, memory) * beta.unsqueeze(-1)
        weights, mask = topk_sparse_softmax(sim, self.topk)
        read = torch.einsum('bhn,bnd->bhd', weights, memory)
        read_combined = self.head_merge(read.reshape(read.shape[0], -1))
        if self.policy == "lru":
            indices = mask.float().argmax(-1)
            self.access_times = self.access_times.scatter_add(1, indices, torch.ones_like(indices, dtype=torch.float32))
        return read_combined, weights

    def write(self, memory: torch.Tensor, write_keys: torch.Tensor, write_vals: torch.Tensor,
              erase: torch.Tensor, add_gate: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = self.cosine_sim(write_keys, memory) * beta.unsqueeze(-1)
        if self.policy == "learned":
            priorities = self.priority_mlp(memory).squeeze(-1).unsqueeze(1)  # (B,1,N)
            sim = sim + priorities
        weights, mask = topk_sparse_softmax(sim, self.topk)
        if self.policy == "lru":
            low_sim_mask = (sim < 0.1).all(-1)  # evict if low sim
            min_access_idx = self.access_times.argmin(dim=-1, keepdim=True).unsqueeze(1).expand(-1, self.n_heads, -1)  # (B,H,1)
            eviction_mask = low_sim_mask.unsqueeze(-1)  # (B,H,1)
            weights = weights.scatter_add(2, min_access_idx, eviction_mask.float())  # boost weight for eviction
            indices = mask.float().argmax(-1)
            self.access_times = self.access_times.scatter_add(1, indices, torch.ones_like(indices, dtype=torch.float32))
        w_unsq = weights.unsqueeze(-1)
        erase_unsq = erase.unsqueeze(-1).unsqueeze(-1)
        add_unsq = add_gate.unsqueeze(-1).unsqueeze(-1) * write_vals.unsqueeze(2)
        mem_exp = memory.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mem_after_erase = mem_exp * (1 - w_unsq * erase_unsq)
        mem_after_add = mem_after_erase + w_unsq * add_unsq
        new_memory = mem_after_add.mean(dim=1)
        new_memory = F.normalize(new_memory + 1e-8, dim=-1)
        return new_memory, weights
