from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def topk_sparse_softmax(sim: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Top-K sparse softmax.
    Values outside top-k are masked to -inf before softmax.
    Returns:
        weights: Softmax distribution over top-k slots [B, H, N]
        mask: Boolean mask indicating selected slots [B, H, N]
    """
    B, H, N = sim.shape
    sim_flat = sim.view(B * H, N)
    
    # Select Top-K indices
    _, idx = torch.topk(sim_flat, k=min(k, N), dim=-1)
    
    # Create mask
    mask = torch.zeros_like(sim_flat, dtype=torch.bool)
    arange = torch.arange(B * H, device=sim.device).unsqueeze(1)
    mask[arange, idx] = True
    mask = mask.view(B, H, N)
    
    # Apply mask and softmax
    masked_sim = sim.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_sim, dim=-1)
    
    return weights, mask

class MemorySynthesizer(nn.Module):
    """
    Implements 'Imaginative Replay'.
    A small Transformer that allows memory slots to attend to each other
    and synthesize new connections/abstractions without external input.
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
        return self.transformer(memory)

class MultiHeadMemoryBank(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, n_heads: int = 8, topk: int = 16,
                 policy: str = "meta", use_decay_gate: bool = True, decay_rate: float = 0.99,
                 bottleneck_dim: int = 64, n_synthesis_layers: int = 2, synthesis_heads: int = 4,
                 use_hebbian_graph: bool = True, hebbian_lr: float = 0.05, 
                 hebbian_decay: float = 0.995, graph_influence: float = 0.2,
                 graph_rank: int = 32, use_lsh: bool = True, hash_buckets: int = 32):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.topk = topk
        self.policy = policy
        
        # Hebbian Graph Parameters
        self.use_hebbian_graph = use_hebbian_graph
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.graph_influence = graph_influence
        self.graph_rank = graph_rank

        # Low-rank adjacency: U [B, Slots, Rank], V [B, Rank, Slots]
        # Effective adjacency = U @ V
        self.register_buffer("U", torch.zeros(1, num_slots, graph_rank))
        self.register_buffer("V", torch.zeros(1, graph_rank, num_slots))
        
        # Buffer to track previous access pattern for temporal association
        self.register_buffer("prev_access_mean", torch.zeros(1, num_slots))

        # Semantic Bottleneck
        # Compresses information before writing to ensure density
        self.bottleneck = nn.Sequential(
            nn.Linear(slot_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, slot_dim)
        )

        # Neural Synthesizer
        self.synthesizer = MemorySynthesizer(slot_dim, synthesis_heads, n_synthesis_layers)

        # Meta-Policy Gate
        # Decides mixture of [TopK, Uniform, Random] strategies
        if policy == "meta":
            self.meta_gate = nn.Linear(slot_dim, 3) 

        # Memory State
        self.init_memory = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.01)
        nn.init.orthogonal_(self.init_memory)
        
        # Track age of slots (LRU logic)
        self.register_buffer("age", torch.zeros(1, num_slots))
        
        # Output projection & Normalization
        self.norm = nn.LayerNorm(slot_dim)
        self.head_merge = nn.Linear(n_heads * slot_dim, slot_dim)

        # Decay mechanism
        self.use_decay_gate = use_decay_gate
        self.decay_rate = decay_rate
        if use_decay_gate:
            self.decay_gate = nn.Parameter(torch.ones(num_slots) * 0.99)
        
        if policy == "learned":
            self.priority_mlp = nn.Linear(slot_dim, 1)
        
        # Sparse Attention with LSH
        self.use_lsh = use_lsh
        self.hash_buckets = hash_buckets
        if use_lsh:
            self.random_projections = nn.Parameter(torch.randn(slot_dim, hash_buckets // 2))

    def reset_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Resets memory states, graph connections, and age buffers."""
        self.age = torch.zeros(batch_size, self.num_slots, device=device)
        self.U = torch.zeros(batch_size, self.num_slots, self.graph_rank, device=device)
        self.V = torch.zeros(batch_size, self.graph_rank, self.num_slots, device=device)
        self.prev_access_mean = torch.zeros(batch_size, self.num_slots, device=device)
        
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
        """
        Runs the 'dreaming' process.
        Slots attend to each other to refine representations.
        """
        delta = self.synthesizer(memory)
        # Residual update with normalization
        return F.normalize(memory + 0.1 * delta, dim=-1)

    def update_hebbian_graph(self, current_weights: torch.Tensor):
        """
        Updates the low-rank factors U and V using a temporal Hebbian rule.
        """
        if not self.use_hebbian_graph:
            return

        with torch.no_grad():
            curr_act = current_weights.mean(dim=1)  # [B, N]
            prev_act = self.prev_access_mean        # [B, N]

            # Outer product approximation for low-rank: hebbian_update ≈ prev_act.unsqueeze(2) @ curr_act.unsqueeze(1)
            # But for low-rank, add to U/V incrementally (approximate)
            delta_U = prev_act.unsqueeze(2) * self.hebbian_lr  # [B, N, 1] but expand to rank if needed;
            delta_V = curr_act.unsqueeze(1) * self.hebbian_lr  # [B, 1, N]
            
            # For full rank, we can accumulate; but to keep rank, use rank-1 update (Woodbury or similar, but simple add)
            self.U = (self.U * self.hebbian_decay) + torch.cat([self.U[:, :, :-1], delta_U], dim=-1) if self.U.shape[-1] > self.graph_rank else self.U + delta_U.expand(-1, -1, self.graph_rank)
            self.V = (self.V * self.hebbian_decay) + torch.cat([self.V[:, :-1, :], delta_V], dim=1) if self.V.shape[1] > self.graph_rank else self.V + delta_V.expand(-1, self.graph_rank, -1)
            
            # Clamp to prevent explosion
            self.U = torch.clamp(self.U, -1.0, 1.0)
            self.V = torch.clamp(self.V, -1.0, 1.0)
            
            self.prev_access_mean = curr_act.detach()

    def read(self, memory: torch.Tensor, read_keys: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads from memory using Meta-Policy and Spreading Activation, with sparse LSH for efficiency.
        """
        # 1. Compute Base Similarity (with optional LSH for sparse)
        if self.use_lsh:
            # LSH: Project to hashes, find buckets
            proj_keys = torch.sign(read_keys @ self.random_projections)  # [B, H, buckets/2]
            proj_mem = torch.sign(memory @ self.random_projections)  # [B, N, buckets/2]
            buckets = (proj_keys.unsqueeze(2) == proj_mem.unsqueeze(1)).float().mean(-1) > 0.5  # Approximate match [B, H, N]
            sim = self.cosine_sim(read_keys, memory) * buckets * beta.unsqueeze(-1)
        else:
            sim = self.cosine_sim(read_keys, memory) * beta.unsqueeze(-1)
        
        w_base, mask = topk_sparse_softmax(sim, self.topk)  # [B, H, N]

        # 2. Hebbian Spreading Activation (using low-rank adjacency)
        if self.use_hebbian_graph:
            # Reconstructed adjacency ≈ self.U @ self.V [B, N, N]
            adjacency_approx = torch.bmm(self.U, self.V)
            
            spread_signal = torch.bmm(w_base, adjacency_approx)  # [B, H, N]
            
            w_combined = w_base + (self.graph_influence * spread_signal)
            w_final = w_combined / (w_combined.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            w_final = w_base

        # 3. Meta-Policy Mixing
        if self.policy == "meta":
            gate_logits = self.meta_gate(read_keys) 
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1) 
            w_uniform = torch.ones_like(w_final) / self.num_slots
            w_random = F.softmax(torch.randn_like(w_final) * 5.0, dim=-1)
            w_stack = torch.stack([w_final, w_uniform, w_random], dim=2)
            final_weights = (w_stack * gate_weights).sum(dim=2)
        else:
            final_weights = w_final

        # 4. Retrieve Content
        read_per_head = torch.einsum('bhn,bnd->bhd', final_weights, memory)
        read_combined = self.head_merge(read_per_head.reshape(read_per_head.shape[0], -1))
        read_combined = self.norm(read_combined)

        # 5. Update Age and Hebbian Graph
        with torch.no_grad():
            accessed_slots = mask.any(dim=1).float()
            self.age = self.age * (1 - accessed_slots) + accessed_slots * 0.0
        self.update_hebbian_graph(final_weights)

        return read_combined, final_weights

    def write(self, memory: torch.Tensor, write_keys: torch.Tensor, write_vals: torch.Tensor,
              erase: torch.Tensor, add_gate: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Increment age for all slots
        with torch.no_grad():
            self.age += 1.0

        # Pass value through Bottleneck
        compressed_vals = self.bottleneck(write_vals)
        
        sim = self.cosine_sim(write_keys, memory) * beta.unsqueeze(-1)

        # LRU Logic: Bias writing towards older/unused slots
        age_bias = (self.age / (self.age.max() + 1e-8)).unsqueeze(1)
        sim = sim + age_bias

        weights, _ = topk_sparse_softmax(sim, self.topk)

        # Standard Memory Update (Erase + Add)
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