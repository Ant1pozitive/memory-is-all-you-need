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
    
    _, idx = torch.topk(sim_flat, k=min(k, N), dim=-1)
    
    mask = torch.zeros_like(sim_flat, dtype=torch.bool)
    arange = torch.arange(B * H, device=sim.device).unsqueeze(1)
    mask[arange, idx] = True
    mask = mask.view(B, H, N)
    
    masked_sim = sim.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_sim, dim=-1)
    
    return weights, mask

class MemorySynthesizer(nn.Module):
    """
    Implements 'Imaginative Replay'.
    """
    def __init__(self, slot_dim: int, n_heads: int, n_layers: int, device: str):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim, nhead=n_heads, dim_feedforward=slot_dim * 2,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(device)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.transformer(memory)


class ContextWeightingNetwork(nn.Module):
    """
    Learns to assign importance weights to memory dimensions based on the query context.
    Input: Query Vector [B, H, D]
    Output: Feature Weights [B, H, D] (in range 0-2, centered at 1)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # Scale sigmoid to [0, 2] to allow both suppression and enhancement
        return self.net(query) * 2.0


class MultiHeadMemoryBank(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, n_heads: int = 8, topk: int = 16,
                 policy: str = "meta", use_decay_gate: bool = True, decay_rate: float = 0.99,
                 bottleneck_dim: int = 64, n_synthesis_layers: int = 2, synthesis_heads: int = 4,
                 use_hebbian_graph: bool = True, hebbian_lr: float = 0.05, 
                 hebbian_decay: float = 0.995, graph_influence: float = 0.3, cfg=None):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.topk = topk
        self.policy = policy
        
        # Hebbian Graph & Topological parameters
        self.use_hebbian_graph = use_hebbian_graph
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.graph_influence = graph_influence
        self.spreading_steps = cfg.memory.spreading_steps if cfg else 1

        # Context-Dependent Metric Learning
        self.use_context_metric = cfg.memory.use_context_metric if cfg else False
        if self.use_context_metric:
            self.context_weigher = ContextWeightingNetwork(slot_dim).to(cfg.device)

        # Semantic Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(slot_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, slot_dim)
        ).to(cfg.device)

        # Neural Synthesizer
        self.synthesizer = MemorySynthesizer(slot_dim, synthesis_heads, n_synthesis_layers, cfg.device)

        # Meta-Policy Gate
        if policy == "meta":
            self.meta_gate = nn.Linear(slot_dim, 3).to(cfg.device) 

        # Base init vector
        self.init_memory = nn.Parameter(torch.randn(1, 1, slot_dim, device=cfg.device) * 0.01)
        nn.init.orthogonal_(self.init_memory)

        # Decay mechanism
        self.use_decay_gate = use_decay_gate
        self.decay_rate = decay_rate
        if use_decay_gate:
            self.decay_gate = nn.Parameter(torch.ones(1, device=cfg.device) * 0.99)
        
        # STM/LTM params
        self.stm_slots = cfg.memory.stm_slots
        self.ltm_slots = cfg.memory.ltm_slots
        self.stm_decay_rate = cfg.memory.stm_decay_rate
        self.ltm_decay_rate = cfg.memory.ltm_decay_rate
        
        # Curriculum Consolidation
        self.use_dynamic_consolidation = cfg.memory.use_dynamic_consolidation
        self.base_consolidation_threshold = cfg.memory.consolidation_threshold
        self.consolidation_percentile = cfg.memory.consolidation_percentile
        
        # Forgetting
        self.prune_threshold = cfg.memory.prune_threshold
        self.forget_rate_multiplier = cfg.memory.forget_rate_multiplier

        # Buffers
        self.register_buffer("stm_adjacency", torch.zeros(1, self.stm_slots, self.stm_slots, device=cfg.device))
        self.register_buffer("stm_prev_access_mean", torch.zeros(1, self.stm_slots, device=cfg.device))
        self.register_buffer("stm_age", torch.zeros(1, self.stm_slots, device=cfg.device))

        self.register_buffer("ltm_adjacency", torch.zeros(1, self.ltm_slots, self.ltm_slots, device=cfg.device))
        self.register_buffer("ltm_prev_access_mean", torch.zeros(1, self.ltm_slots, device=cfg.device))
        self.register_buffer("ltm_age", torch.zeros(1, self.ltm_slots, device=cfg.device))

    def reset_memory(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stm_age = torch.zeros(batch_size, self.stm_slots, device=device)
        self.stm_adjacency = torch.zeros(batch_size, self.stm_slots, self.stm_slots, device=device)
        self.stm_prev_access_mean = torch.zeros(batch_size, self.stm_slots, device=device)
        stm_mem = self.init_memory.expand(batch_size, self.stm_slots, -1).clone().to(device)
        
        self.ltm_age = torch.zeros(batch_size, self.ltm_slots, device=device)
        self.ltm_adjacency = torch.zeros(batch_size, self.ltm_slots, self.ltm_slots, device=device)
        self.ltm_prev_access_mean = torch.zeros(batch_size, self.ltm_slots, device=device)
        ltm_mem = self.init_memory.expand(batch_size, self.ltm_slots, -1).clone().to(device)
        
        return stm_mem, ltm_mem

    def apply_decay(self, memory: torch.Tensor, decay_rate: float) -> torch.Tensor:
        if self.use_decay_gate:
            decay = torch.sigmoid(self.decay_gate).view(1, 1, 1)
            memory = memory * decay
        else:
            memory = memory * decay_rate
        return memory

    def synthesize(self, memory: torch.Tensor) -> torch.Tensor:
        delta = self.synthesizer(memory)
        return F.normalize(memory + 0.1 * delta, dim=-1)

    def update_hebbian_graph(self, current_weights: torch.Tensor, adjacency: torch.Tensor, prev_access_mean: torch.Tensor):
        if not self.use_hebbian_graph:
            return adjacency, prev_access_mean

        with torch.no_grad():
            curr_act = current_weights.mean(dim=1)
            hebbian_update = torch.bmm(prev_access_mean.unsqueeze(2), curr_act.unsqueeze(1))
            adjacency = (adjacency * self.hebbian_decay) + (self.hebbian_lr * hebbian_update)
            adjacency = torch.clamp(adjacency, max=1.0)
            prev_access_mean = curr_act.detach()
        
        return adjacency, prev_access_mean

    def read(self, memory: torch.Tensor, adjacency: torch.Tensor, read_keys: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Context-Dependent Metric Learning
        # Instead of static cosine similarity, we weight features dynamically based on the query.
        if self.use_context_metric:
            # Generate feature importance weights: [B, H, D]
            feature_weights = self.context_weigher(read_keys)
            
            # Apply weights to query keys: [B, H, D]
            weighted_keys = read_keys * feature_weights
            
            # Apply weights to memory: [B, 1, N, D] * [B, H, 1, D] -> [B, H, N, D]
            # We treat each head's context differently for the same memory slots
            weighted_memory = memory.unsqueeze(1) * feature_weights.unsqueeze(2)
            
            # Compute Cosine Sim on weighted vectors
            k_norm = F.normalize(weighted_keys, dim=-1)
            m_norm = F.normalize(weighted_memory, dim=-1)
            sim = torch.einsum('bhd,bhnd->bhn', k_norm, m_norm) * beta.unsqueeze(-1)
        else:
            # Standard Cosine
            k_norm = F.normalize(read_keys, dim=-1)
            m_norm = F.normalize(memory, dim=-1)
            sim = torch.einsum('bhd,bnd->bhn', k_norm, m_norm) * beta.unsqueeze(-1)

        # 2. Content Addressing (Top-K)
        w_content, _ = topk_sparse_softmax(sim, self.topk)

        # 3. Topological Memory: Spreading Activation
        # Activation flows through the graph: w(t+1) = w(t) * Adjacency
        if self.use_hebbian_graph:
            w_current = w_content
            for _ in range(self.spreading_steps):
                # Spread: [B, H, N] x [B, N, N] -> [B, H, N]
                spread_signal = torch.matmul(w_current, adjacency)
                # Combine original activation with spread signal
                w_current = w_current + (self.graph_influence * spread_signal)
            
            # Re-normalize to ensure valid probability distribution
            w_final = w_current / (w_current.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            w_final = w_content

        # 4. Meta-Policy
        if self.policy == "meta":
            gate_logits = self.meta_gate(read_keys)
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)

            w_uniform = torch.ones_like(w_final) / memory.size(1)
            w_random = F.softmax(torch.randn_like(w_final) * 5.0, dim=-1)

            w_stack = torch.stack([w_final, w_uniform, w_random], dim=2)
            final_weights = (w_stack * gate_weights).sum(dim=2)
        else:
            final_weights = w_final

        # 5. Retrieve
        read_per_head = torch.einsum('bhn,bnd->bhd', final_weights, memory)
        head_merge = nn.Linear(self.n_heads * self.slot_dim, self.slot_dim, device=memory.device)
        norm = nn.LayerNorm(self.slot_dim, device=memory.device)
        read_combined = norm(head_merge(read_per_head.reshape(read_per_head.shape[0], -1)))

        return read_combined, final_weights

    def write(self, memory: torch.Tensor, adjacency: torch.Tensor, age: torch.Tensor, prev_access_mean: torch.Tensor,
              write_keys: torch.Tensor, write_vals: torch.Tensor, erase: torch.Tensor, add_gate: torch.Tensor, beta: torch.Tensor, decay_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            age += 1.0

        compressed_vals = self.bottleneck(write_vals)
        
        # Apply metric learning to write keys too for consistency
        if self.use_context_metric:
            feature_weights = self.context_weigher(write_keys)
            weighted_keys = write_keys * feature_weights
            weighted_memory = memory.unsqueeze(1) * feature_weights.unsqueeze(2)
            
            k_norm = F.normalize(weighted_keys, dim=-1)
            m_norm = F.normalize(weighted_memory, dim=-1)
            sim = torch.einsum('bhd,bhnd->bhn', k_norm, m_norm) * beta.unsqueeze(-1)
        else:
            k_norm = F.normalize(write_keys, dim=-1)
            m_norm = F.normalize(memory, dim=-1)
            sim = torch.einsum('bhd,bnd->bhn', k_norm, m_norm) * beta.unsqueeze(-1)

        age_bias = (age / (age.max() + 1e-8)).unsqueeze(1)
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
        new_memory = self.apply_decay(new_memory, decay_rate)

        with torch.no_grad():
            accessed = (weights.mean(dim=1) > 0.01).float()
            age = age * (1 - accessed)

        return new_memory, weights

    def consolidate(self, stm_memory, stm_weights, ltm_memory, ltm_adjacency):
        with torch.no_grad():
            stm_util = stm_weights.mean(dim=1).mean(dim=0)
            
            if self.use_dynamic_consolidation:
                threshold = torch.quantile(stm_util, self.consolidation_percentile)
                threshold = max(threshold, self.base_consolidation_threshold)
            else:
                threshold = self.base_consolidation_threshold
            
            candidate_mask = stm_util > threshold
            if not candidate_mask.any():
                return ltm_memory, ltm_adjacency

            src_indices = torch.nonzero(candidate_mask).squeeze(1)
            
            if len(src_indices) > 16:
                 _, top_k = torch.topk(stm_util, k=16)
                 src_indices = top_k

            ltm_age_mean = self.ltm_age.mean(dim=0)
            _, dst_indices = torch.topk(ltm_age_mean, k=len(src_indices))
            
            transferred_content = stm_memory[:, src_indices, :]
            ltm_memory[:, dst_indices, :] = transferred_content
            
            noise = torch.randn_like(transferred_content) * 0.01
            stm_memory[:, src_indices, :] = noise
            self.stm_age[:, src_indices] = 0
            
            self.ltm_age[:, dst_indices] = 0
            
        return ltm_memory, ltm_adjacency

    def prune_weak_edges(self, adjacency: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mask = adjacency > self.prune_threshold
            adjacency = adjacency * mask.float()
        return adjacency

    def active_forget(self, memory: torch.Tensor, forget_mask: torch.Tensor, decay_rate: float) -> torch.Tensor:
        enhanced_decay = decay_rate * (1.0 / self.forget_rate_multiplier)
        decay_tensor = torch.ones_like(forget_mask) * decay_rate
        decay_tensor[forget_mask.bool()] = enhanced_decay
        return memory * decay_tensor.unsqueeze(-1)