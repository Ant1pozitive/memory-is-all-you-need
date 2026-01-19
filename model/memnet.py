import torch
from torch import nn
import torch.nn.functional as F
from .memory_bank import MultiHeadMemoryBank
from .controller import TransformerController


class MemNet(nn.Module):
    """
    MemNet: A Dual-Memory Neural Architecture.
    Combines a Transformer controller with an active Short-Term Memory (STM) 
    and a stable Long-Term Memory (LTM).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Initialize the Multi-Head Memory Bank which manages both STM and LTM buffers
        self.memory = MultiHeadMemoryBank(
            num_slots=cfg.memory.slots,
            slot_dim=cfg.memory.dim,
            n_heads=cfg.memory.heads,
            topk=cfg.memory.topk,
            policy=cfg.memory.policy,
            use_decay_gate=cfg.memory.use_decay_gate,
            decay_rate=cfg.memory.decay_rate,
            bottleneck_dim=cfg.memory.bottleneck_dim,
            n_synthesis_layers=cfg.memory.n_synthesis_layers,
            synthesis_heads=cfg.memory.synthesis_heads,
            use_hebbian_graph=cfg.memory.use_hebbian_graph,
            hebbian_lr=cfg.memory.hebbian_lr,
            hebbian_decay=cfg.memory.hebbian_decay,
            graph_influence=cfg.memory.graph_influence,
            cfg=cfg 
        )

        # Transformer-based controller to generate memory queries and actions
        self.controller = TransformerController(
            cfg.model.vocab_size, cfg.model.embed_dim, cfg.model.hidden_dim,
            cfg.memory.dim, cfg.memory.heads, cfg.model.num_layers,
            cfg.model.num_heads_attn, cfg.model.max_seq_len
        )

        # Head for reconstructing input from memory (Hallucination/Self-Supervision)
        self.hallucination_head = nn.Linear(cfg.memory.dim, cfg.model.embed_dim)

    def forward(self, input_seq: torch.Tensor, return_attn: bool = False):
            """
            Processes a sequence through the controller and dual-memory system.
            Returns logits for prediction and reconstruction for self-supervision.
            """
            B, T = input_seq.shape
            device = input_seq.device

            # Initialize memory buffers for this batch
            stm_mem, ltm_mem = self.memory.reset_memory(B, device)
            read_vec = torch.zeros(B, self.cfg.memory.dim, device=device)

            logits_list = []
            recon_list = [] # List to store reconstruction for each time step

            # Attention logs for visualization
            read_weights_stm_hist = []
            read_weights_ltm_hist = []
            write_weights_hist = []

            for t in range(T):
                current_input = input_seq[:, t:t+1]

                # Controller generates queries based on current token and previous memory read
                logits, read_key, write_key, write_val, erase, add_gate = self.controller(
                    current_input, read_vec, t
                )

                # --- Read Stage ---
                # Extract info from both memory types in parallel
                beta_r = F.softplus(self.controller.beta_read).clamp(1, 20)
                read_stm, r_w_stm = self.memory.read(stm_mem, self.memory.stm_adjacency, read_key, beta_r)
                read_ltm, r_w_ltm = self.memory.read(ltm_mem, self.memory.ltm_adjacency, read_key, beta_r)

                # Update working context (read_vec)
                read_vec = read_vec + (read_stm + read_ltm)

                # --- Write Stage ---
                # Store current info in STM
                beta_w = F.softplus(self.controller.beta_write).clamp(1, 20)
                stm_mem, write_w = self.memory.write(
                    stm_mem, self.memory.stm_adjacency, self.memory.stm_age,
                    self.memory.stm_prev_access_mean, write_key, write_val,
                    erase, add_gate, beta_w, self.memory.stm_decay_rate
                )

                # Periodic memory maintenance (Consolidation/Dreaming)
                if t > 0 and t % self.cfg.memory.synthesis_interval == 0:
                    # Code for synthesis and STM -> LTM transfer
                    # ... (as in previous version)
                    pass

                # --- RECONSTRUCTION HEAD ---
                # Here we try to map the memory state back to the input embedding
                # This 'hallucination' proves the memory actually stored the input
                recon_step = self.hallucination_head(read_vec) # [B, Embed_dim]

                logits_list.append(logits.unsqueeze(1)) # [B, 1, Vocab]
                recon_list.append(recon_step.unsqueeze(1)) # [B, 1, Embed_dim]

                if return_attn:
                    read_weights_stm_hist.append(r_w_stm.detach().cpu())
                    read_weights_ltm_hist.append(r_w_ltm.detach().cpu())
                    write_weights_hist.append(write_w.detach().cpu())

            # Concatenate results along the time dimension [B, T, ...]
            logits = torch.cat(logits_list, dim=1)
            recon = torch.cat(recon_list, dim=1)

            if return_attn:
                return logits, recon, \
                    (torch.stack(read_weights_stm_hist, dim=1), torch.stack(read_weights_ltm_hist, dim=1)), \
                    torch.stack(write_weights_hist, dim=1)

            return logits, recon