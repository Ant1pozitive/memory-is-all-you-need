"""Multi-head Memory Bank with top-k sparse addressing.

Memory shape: (B, N, D)
Read/write keys: (B, H, D)
We compute similarity per head -> (B, H, N), then select top-k per head and softmax over those to produce sparse weights.

Writes aggregate contributions from heads.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn

def topk_sparse_softmax(sim: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    sim: (B, H, N)
    returns: (weights, mask) where weights (B,H,N) has softmax only over top-k indices, mask is boolean topk mask
    """
    B, H, N = sim.shape
    sim_flat = sim.view(B * H, N)
    vals, idx = torch.topk(sim_flat, k=min(k, N), dim=-1)
    # create mask
    mask = torch.zeros_like(sim_flat, dtype=torch.bool)
    arange = torch.arange(B * H, device=sim.device).unsqueeze(1)
    mask[arange, idx] = True
    mask = mask.view(B, H, N)
    # for numerical stability, set non-topk to -inf so softmax zero them
    neg_inf = -1e9
    masked = sim.masked_fill(~mask, neg_inf)
    weights = F.softmax(masked, dim=-1)
    return weights, mask

class MultiHeadMemoryBank(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, n_heads: int = 4, topk: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.topk = topk

        # learnable initial memory bias
        self.init_memory = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.01)

    def reset_memory(self, batch_size: int, device):
        return self.init_memory.expand(batch_size, -1, -1).contiguous().to(device)

    @staticmethod
    def cosine_sim(keys: torch.Tensor, memory: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # keys: (B, H, D), memory: (B, N, D) -> sim: (B, H, N)
        B, H, D = keys.shape
        N = memory.shape[1]
        # normalize
        k_norm = keys.norm(p=2, dim=-1, keepdim=True).clamp_min(eps) # (B,H,1)
        m_norm = memory.norm(p=2, dim=-1).clamp_min(eps) # (B,N)
        # compute dot: expand memory to (B,1,N,D) and keys to (B,H,1,D)
        dot = torch.einsum('bhd,bnd->bhn', keys, memory)
        sim = dot / (k_norm.squeeze(-1).unsqueeze(-1) * m_norm.unsqueeze(1) + eps)
        return sim

    def read(self, memory: torch.Tensor, read_keys: torch.Tensor, beta: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read with multi-head top-k sparse addressing.
        read_keys: (B, H, D)
        memory: (B, N, D)
        returns: read_vec (B, H, D) aggregated per head (or combined to B,D), weights (B,H,N)
        """
        sim = self.cosine_sim(read_keys, memory) # (B,H,N)
        sim = sim * beta
        weights, mask = topk_sparse_softmax(sim, self.topk) # (B,H,N)
        # read per head
        read = torch.einsum('bhn,bnd->bhd', weights, memory) # (B,H,D)
        # optionally combine heads by averaging
        read_combined = read.mean(dim=1) # (B,D)
        return read_combined, weights

    def write(self, memory: torch.Tensor, write_keys: torch.Tensor, write_vals: torch.Tensor,
    erase: torch.Tensor = None, beta: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write with multi-head addressing.
        write_keys: (B,H,D), write_vals: (B,H,D) -> we compute per-head weights and apply
        returns updated memory and the write weights (B,H,N)
        """
        sim = self.cosine_sim(write_keys, memory) # (B,H,N)
        sim = sim * beta
        weights, mask = topk_sparse_softmax(sim, self.topk) # (B,H,N)


        B, H, N = weights.shape
        _, _, D = write_vals.shape
        w_unsq = weights.unsqueeze(-1) # (B,H,N,1)
        # prepare erase and add
        if erase is None:
            erase = torch.zeros_like(write_vals) # (B,H,D)
        # memory shape: (B,N,D) -> expand to (B,H,N,D)
        mem_exp = memory.unsqueeze(1).expand(-1, H, -1, -1)
        # apply erase per head: M = M * (1 - w * erase)
        erase_unsq = erase.unsqueeze(2) # (B,H,1,D)
        mem_after_erase = mem_exp * (1 - w_unsq * erase_unsq)
        # apply add: add contribution w * write_val
        add_unsq = write_vals.unsqueeze(2) # (B,H,1,D)
        mem_after_add = mem_after_erase + w_unsq * add_unsq
        # aggregate heads by summing across heads
        new_memory = mem_after_add.sum(dim=1) # (B,N,D)
        return new_memory, weights
