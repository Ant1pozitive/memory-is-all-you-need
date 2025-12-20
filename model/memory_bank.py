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

class MemorySynthesizer(nn.Module):
    """
    Implements 'Imaginative Replay': Allows memory slots to attend to each other
    and synthesize new abstract representations without external input.
    """
    def __init__(self, slot_dim: int, n_heads: int, n_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim, nhead=n_heads, dim_feedforward=slot_dim * 2,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        # memory: [B, Slots, Dim]
        # Self-attention over slots (inter-slot communication)
        refined_memory = self.transformer(memory)
        return refined_memory

class MultiHeadMemoryBank(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, n_heads: int = 8, topk: int = 16,
                 policy: str = "meta", use_decay_gate: bool = True, decay_rate: float = 0.99,
                 bottleneck_dim: int = 64, n_synthesis_layers: int = 2, synthesis_heads: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.topk = topk
        self.policy = policy
        self.use_decay_gate = use_decay_gate
        self.decay_rate = decay_rate

        # Semantic Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(slot_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, slot_dim)
        )

        # Neural Memory Synthesizer (The "Dreaming" Module)
        self.synthesizer = MemorySynthesizer(slot_dim, synthesis_heads, n_synthesis_layers)

        # Meta-Policy Gates: [TopK, Uniform, Random]
        if policy == "meta":
            self.meta_gate = nn.Linear(slot_dim, 3) 

        self.init_memory = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.01)
        nn.init.orthogonal_(self.init_memory)

        self.register_buffer("age", torch.zeros(1, num_slots))
        
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

    def synthesize(self, memory: torch.Tensor) -> torch.Tensor:
        """Runs the internal synthesis process (residual update)"""
        delta = self.synthesizer(memory)
        # Residual connection: Memory = Memory + Synthesis
        return F.normalize(memory + 0.1 * delta, dim=-1)

    def read(self, memory: torch.Tensor, read_keys: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = self.cosine_sim(read_keys, memory) * beta.unsqueeze(-1)
        
        # Base Top-K weights
        w_topk, mask = topk_sparse_softmax(sim, self.topk)

        if self.policy == "meta":
            # Compute mixing coefficients based on Query
            # gate_logits: [B, H, 3]
            gate_logits = self.meta_gate(read_keys) 
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1) # [B, H, 3, 1]

            # 1. Top-K Strategy
            # 2. Uniform Strategy (Exploration)
            w_uniform = torch.ones_like(w_topk) / self.num_slots
            # 3. Random Strategy (Noise/Dream)
            w_random = F.softmax(torch.randn_like(sim), dim=-1)

            # Mix policies
            # Stack: [B, H, 3, N]
            w_stack = torch.stack([w_topk, w_uniform, w_random], dim=2)
            final_weights = (w_stack * gate_weights).sum(dim=2) # Weighted sum
        else:
            final_weights = w_topk

        read_per_head = torch.einsum('bhn,bnd->bhd', final_weights, memory)
        read_combined = self.head_merge(read_per_head.reshape(read_per_head.shape[0], -1))
        
        read_combined = self.norm(read_combined)

        with torch.no_grad():
            accessed_slots = mask.any(dim=1).float()
            self.age = self.age * (1 - accessed_slots) + accessed_slots * 0.0

        return read_combined, final_weights

    def write(self, memory: torch.Tensor, write_keys: torch.Tensor, write_vals: torch.Tensor,
              erase: torch.Tensor, add_gate: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            self.age += 1.0

        compressed_vals = self.bottleneck(write_vals)
        sim = self.cosine_sim(write_keys, memory) * beta.unsqueeze(-1)

        # LRU Bias
        age_bias = (self.age / (self.age.max() + 1e-8)).unsqueeze(1)
        sim = sim + age_bias

        weights, _ = topk_sparse_softmax(sim, self.topk)

        w_unsq = weights.unsqueeze(-1)
        erase_unsq = erase.unsqueeze(-1).unsqueeze(-1)
        add_unsq = add_gate.unsqueeze(-1).unsqueeze(-1) * compressed_vals.unsqueeze(2)

        mem_exp = memory.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mem_after_erase = mem_exp * (1 - w_unsq * erase_unsq)
        
        new_memory = mem_after_erase + w_unsq * add_unsq
        new_memory = new_memory.mean(dim=1)
        
        new_memory = F.normalize(new_memory + 1e-8, dim=-1)
        new_memory = self.apply_decay(new_memory)

        return new_memory, weights