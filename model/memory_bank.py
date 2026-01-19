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
    Implements 'Imaginative Replay' / Dreaming.
    A small Transformer that allows memory slots to attend to each other
    and synthesize new connections/abstractions without external input.
    """
    def __init__(self, slot_dim: int, n_heads: int, n_layers: int, device: str):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim, nhead=n_heads, dim_feedforward=slot_dim * 2,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(device)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        # memory: [B, Slots, Dim]
        return self.transformer(memory)

class MultiHeadMemoryBank(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, n_heads: int = 8, topk: int = 16,
                 policy: str = "meta", use_decay_gate: bool = True, decay_rate: float = 0.99,
                 bottleneck_dim: int = 64, n_synthesis_layers: int = 2, synthesis_heads: int = 4,
                 use_hebbian_graph: bool = True, hebbian_lr: float = 0.05, 
                 hebbian_decay: float = 0.995, graph_influence: float = 0.2, cfg=None):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.topk = topk
        self.policy = policy
        
        # Hebbian Graph Parameters
        self.use_hebbian_graph = use_hebbian_graph
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.graph_influence = graph_influence

        # Semantic Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(slot_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, slot_dim)
        ).to(cfg.device)

        # Neural Synthesizer (Dreaming)
        self.synthesizer = MemorySynthesizer(slot_dim, synthesis_heads, n_synthesis_layers, cfg.device)

        # Meta-Policy Gate
        if policy == "meta":
            self.meta_gate = nn.Linear(slot_dim, 3).to(cfg.device) 

        # Base init vector for slots
        self.init_memory = nn.Parameter(torch.randn(1, 1, slot_dim, device=cfg.device) * 0.01)
        nn.init.orthogonal_(self.init_memory)

        # Decay mechanism
        self.use_decay_gate = use_decay_gate
        self.decay_rate = decay_rate
        if use_decay_gate:
            self.decay_gate = nn.Parameter(torch.ones(1, device=cfg.device) * 0.99)
        
        # STM/LTM params
        self.stm_slots = cfg.memory.stm_slots if cfg else 64
        self.ltm_slots = cfg.memory.ltm_slots if cfg else 1024
        self.stm_decay_rate = cfg.memory.stm_decay_rate if cfg else 0.95
        self.ltm_decay_rate = cfg.memory.ltm_decay_rate if cfg else 0.995
        self.consolidation_threshold = cfg.memory.consolidation_threshold if cfg else 0.8
        
        # Forgetting params
        self.prune_threshold = cfg.memory.prune_threshold if cfg else 0.1
        self.forget_rate_multiplier = cfg.memory.forget_rate_multiplier if cfg else 2.0

        # STM buffers
        self.register_buffer("stm_adjacency", torch.zeros(1, self.stm_slots, self.stm_slots, device=cfg.device))
        self.register_buffer("stm_prev_access_mean", torch.zeros(1, self.stm_slots, device=cfg.device))
        self.register_buffer("stm_age", torch.zeros(1, self.stm_slots, device=cfg.device))

        # LTM buffers
        self.register_buffer("ltm_adjacency", torch.zeros(1, self.ltm_slots, self.ltm_slots, device=cfg.device))
        self.register_buffer("ltm_prev_access_mean", torch.zeros(1, self.ltm_slots, device=cfg.device))
        self.register_buffer("ltm_age", torch.zeros(1, self.ltm_slots, device=cfg.device))

    def reset_memory(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resets STM and LTM states, graph connections, and age buffers."""
        # STM reset
        self.stm_age = torch.zeros(batch_size, self.stm_slots, device=device)
        self.stm_adjacency = torch.zeros(batch_size, self.stm_slots, self.stm_slots, device=device)
        self.stm_prev_access_mean = torch.zeros(batch_size, self.stm_slots, device=device)
        stm_mem = self.init_memory.expand(batch_size, self.stm_slots, -1).clone().to(device)
        
        # LTM reset
        self.ltm_age = torch.zeros(batch_size, self.ltm_slots, device=device)
        self.ltm_adjacency = torch.zeros(batch_size, self.ltm_slots, self.ltm_slots, device=device)
        self.ltm_prev_access_mean = torch.zeros(batch_size, self.ltm_slots, device=device)
        ltm_mem = self.init_memory.expand(batch_size, self.ltm_slots, -1).clone().to(device)
        
        return stm_mem, ltm_mem

    @staticmethod
    def cosine_sim(keys: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        k_norm = F.normalize(keys, dim=-1)
        m_norm = F.normalize(memory, dim=-1)
        return torch.einsum('bhd,bnd->bhn', k_norm, m_norm)

    def apply_decay(self, memory: torch.Tensor, decay_rate: float) -> torch.Tensor:
        if self.use_decay_gate:
            decay = torch.sigmoid(self.decay_gate).view(1, 1, 1)
            memory = memory * decay
        else:
            memory = memory * decay_rate
        return memory

    def synthesize(self, memory: torch.Tensor) -> torch.Tensor:
        """Run dreaming/synthesis on a memory block"""
        delta = self.synthesizer(memory)
        # Residual update with normalization
        return F.normalize(memory + 0.1 * delta, dim=-1)

    def update_hebbian_graph(self, current_weights: torch.Tensor, adjacency: torch.Tensor, prev_access_mean: torch.Tensor):
        if not self.use_hebbian_graph:
            return adjacency, prev_access_mean

        with torch.no_grad():
            # Average attention over heads: [B, N]
            curr_act = current_weights.mean(dim=1)
            
            # STDP-like update: Outer product of Prev x Curr
            # [B, N, 1] x [B, 1, N] -> [B, N, N]
            hebbian_update = torch.bmm(prev_access_mean.unsqueeze(2), curr_act.unsqueeze(1))
            
            # Decay old edges and add new associations
            adjacency = (adjacency * self.hebbian_decay) + (self.hebbian_lr * hebbian_update)
            adjacency = torch.clamp(adjacency, max=1.0)
            
            # Update trace
            prev_access_mean = curr_act.detach()
        
        return adjacency, prev_access_mean

    def read(self, memory: torch.Tensor, adjacency: torch.Tensor, read_keys: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Similarity
        sim = self.cosine_sim(read_keys, memory) * beta.unsqueeze(-1)
        w_base, mask = topk_sparse_softmax(sim, self.topk)

        # 2. Spreading Activation via Graph
        if self.use_hebbian_graph:
            # Spread: [B, H, N] x [B, N, N] -> [B, H, N]
            spread_signal = torch.matmul(w_base, adjacency)
            w_combined = w_base + (self.graph_influence * spread_signal)
            w_final = w_combined / (w_combined.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            w_final = w_base

        # 3. Meta-Policy Mixing
        if self.policy == "meta":
            gate_logits = self.meta_gate(read_keys) # [B, H, 3]
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)

            w_uniform = torch.ones_like(w_final) / memory.size(1)
            w_random = F.softmax(torch.randn_like(w_final) * 5.0, dim=-1)

            w_stack = torch.stack([w_final, w_uniform, w_random], dim=2) # [B, H, 3, N]
            final_weights = (w_stack * gate_weights).sum(dim=2)
        else:
            final_weights = w_final

        # 4. Read Content
        read_per_head = torch.einsum('bhn,bnd->bhd', final_weights, memory)
        
        # 5. Merge Heads
        head_merge = nn.Linear(self.n_heads * self.slot_dim, self.slot_dim, device=memory.device)
        norm = nn.LayerNorm(self.slot_dim, device=memory.device)
        read_combined = norm(head_merge(read_per_head.reshape(read_per_head.shape[0], -1)))

        return read_combined, final_weights

    def write(self, memory: torch.Tensor, adjacency: torch.Tensor, age: torch.Tensor, prev_access_mean: torch.Tensor,
              write_keys: torch.Tensor, write_vals: torch.Tensor, erase: torch.Tensor, add_gate: torch.Tensor, beta: torch.Tensor, decay_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            age += 1.0

        # Semantic Bottleneck compression
        compressed_vals = self.bottleneck(write_vals)
        
        # Addressing
        sim = self.cosine_sim(write_keys, memory) * beta.unsqueeze(-1)
        
        # Bias towards older slots (optional LRU-like mechanism via age)
        age_bias = (age / (age.max() + 1e-8)).unsqueeze(1)
        sim = sim + age_bias

        weights, _ = topk_sparse_softmax(sim, self.topk)

        # Write Operation: Erase + Add
        w_unsq = weights.unsqueeze(-1)
        erase_unsq = erase.unsqueeze(-1).unsqueeze(-1)
        add_unsq = add_gate.unsqueeze(-1).unsqueeze(-1) * compressed_vals.unsqueeze(2)

        mem_exp = memory.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mem_after_erase = mem_exp * (1 - w_unsq * erase_unsq)
        
        new_memory = mem_after_erase + w_unsq * add_unsq
        new_memory = new_memory.mean(dim=1) # Average updates from heads
        
        # Normalize and Decay
        new_memory = F.normalize(new_memory + 1e-8, dim=-1)
        new_memory = self.apply_decay(new_memory, decay_rate)

        # Update Age: Reset age for written slots
        with torch.no_grad():
             # Approximation: if weight > 0.01, we touched it
            accessed = (weights.mean(dim=1) > 0.01).float()
            age = age * (1 - accessed)

        return new_memory, weights

    def consolidate(self, stm_memory, stm_weights, ltm_memory, ltm_adjacency):
        """
        Transfers highly utilized STM slots to LTM.
        Replaces Least Recently Used (High Age) slots in LTM.
        """
        with torch.no_grad():
            # 1. Identify important STM slots (mean attention across heads and batch)
            stm_util = stm_weights.mean(dim=1).mean(dim=0) # [STM_Slots]
            
            # Select candidates for consolidation
            # E.g., take top k slots where usage > threshold
            candidate_mask = stm_util > self.consolidation_threshold
            if not candidate_mask.any():
                return ltm_memory, ltm_adjacency

            # Get indices of important STM slots
            src_indices = torch.nonzero(candidate_mask).squeeze(1)
            # Limit to a batch of transfers to avoid overwriting everything
            if len(src_indices) > 16:
                 _, top_k = torch.topk(stm_util, k=16)
                 src_indices = top_k

            # 2. Identify LTM targets (Least Recently Used / Oldest)
            # We use ltm_age to find oldest slots
            ltm_age_mean = self.ltm_age.mean(dim=0)
            _, dst_indices = torch.topk(ltm_age_mean, k=len(src_indices))
            
            # 3. Transfer Content
            # We need to transfer for each batch item
            # src: [B, n_trans, D], dst: [B, n_trans, D]
            transferred_content = stm_memory[:, src_indices, :]
            ltm_memory[:, dst_indices, :] = transferred_content
            
            # 4. Reset Source STM slots (Simulate move by clearing STM)
            # Reset to noise to allow new info
            noise = torch.randn_like(transferred_content) * 0.01
            stm_memory[:, src_indices, :] = noise
            self.stm_age[:, src_indices] = 0
            
            # 5. Reset LTM Age for new entries
            self.ltm_age[:, dst_indices] = 0
            
            # Optional: We could also transfer adjacency, but mapping indices is complex.
            # For now, we let LTM relearn associations for new slots.
            
        return ltm_memory, ltm_adjacency

    def prune_weak_edges(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Explicit Forgetting: Prune weak connections in the Hebbian graph"""
        with torch.no_grad():
            mask = adjacency > self.prune_threshold
            adjacency = adjacency * mask.float()
        return adjacency

    def active_forget(self, memory: torch.Tensor, forget_mask: torch.Tensor, decay_rate: float) -> torch.Tensor:
        """
        Explicit Forgetting: Actively decay specific memory slots.
        forget_mask: [B, Slots], 1 = forget, 0 = keep
        """
        enhanced_decay = decay_rate * (1.0 / self.forget_rate_multiplier) # Lower value = faster decay
        
        # Apply normal decay to kept slots, enhanced decay to forgotten slots
        decay_tensor = torch.ones_like(forget_mask) * decay_rate
        decay_tensor[forget_mask.bool()] = enhanced_decay
        
        return memory * decay_tensor.unsqueeze(-1)