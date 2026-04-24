import torch
from torch import nn
import torch.nn.functional as F
from .memory_bank import MultiHeadMemoryBank
from .controller import TransformerController

class MemNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

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

        self.controller = TransformerController(
            cfg.model.vocab_size, cfg.model.embed_dim, cfg.model.hidden_dim,
            cfg.memory.dim, cfg.memory.heads, cfg.model.num_layers,
            cfg.model.num_heads_attn, cfg.model.max_seq_len
        )

        self.hallucination_head = nn.Linear(cfg.memory.dim, cfg.model.embed_dim)

        # torch.compile compatibility
        if cfg.train.use_compile and torch.__version__ >= "2.0":
            self = torch.compile(self, mode="reduce-overhead")

    def forward(self, input_seq: torch.Tensor, return_attn: bool = False):
        B, T = input_seq.shape
        device = input_seq.device

        stm_mem, ltm_mem = self.memory.reset_memory(B, device)
        read_vec = torch.zeros(B, self.cfg.memory.dim, device=device)

        logits_list = []
        recon_list = []
        
        # Collect for train.py losses
        read_w_stm_list = []
        read_w_ltm_list = []
        write_w_list = []
        adj_stm_list = []
        adj_ltm_list = []

        for t in range(T):
            current_input = input_seq[:, t:t+1]

            logits, read_key, write_key, write_val, erase, add_gate = self.controller(
                current_input, read_vec, t
            )

            # 1. READ with Uncertainty
            beta_r = F.softplus(self.controller.beta_read).clamp(1, 20)
            read_stm, r_w_stm, conf_stm = self.memory.read(
                stm_mem, self.memory.stm_adjacency, read_key, beta_r
            )
            read_ltm, r_w_ltm, conf_ltm = self.memory.read(
                ltm_mem, self.memory.ltm_adjacency, read_key, beta_r
            )

            # Uncertainty-aware fusion
            conf = (conf_stm + conf_ltm) / 2
            read_vec = conf.unsqueeze(1) * read_stm + (1 - conf.unsqueeze(1)) * read_ltm

            # 2. WRITE
            beta_w = F.softplus(self.controller.beta_write).clamp(1, 20)
            stm_mem, write_w = self.memory.write(
                stm_mem, self.memory.stm_adjacency, self.memory.stm_age,
                self.memory.stm_prev_access_mean, write_key, write_val,
                erase, add_gate, beta_w, self.memory.stm_decay_rate
            )

            # Update Graphs
            self.memory.stm_adjacency, self.memory.stm_prev_access_mean = self.memory.update_hebbian_graph(
                write_w, self.memory.stm_adjacency, self.memory.stm_prev_access_mean
            )
            self.memory.ltm_adjacency, self.memory.ltm_prev_access_mean = self.memory.update_hebbian_graph(
                r_w_ltm, self.memory.ltm_adjacency, self.memory.ltm_prev_access_mean
            )

            # 3. CONSOLIDATION / DREAMING
            if t > 0 and t % self.cfg.memory.synthesis_interval == 0:
                stm_mem = self.memory.synthesize(stm_mem)
                ltm_mem, self.memory.ltm_adjacency = self.memory.consolidate(
                    stm_mem, r_w_stm, ltm_mem, self.memory.ltm_adjacency
                )
                self.memory.stm_adjacency = self.memory.prune_weak_edges(self.memory.stm_adjacency)
                self.memory.ltm_adjacency = self.memory.prune_weak_edges(self.memory.ltm_adjacency)

                forget_mask = torch.rand(B, self.cfg.memory.stm_slots, device=device) < 0.05
                stm_mem = self.memory.active_forget(stm_mem, forget_mask, self.memory.stm_decay_rate)
                ltm_mem = self.memory.apply_decay(ltm_mem, self.memory.ltm_decay_rate)

            # 4. RECONSTRUCTION
            recon_vec = self.hallucination_head(read_vec)

            logits_list.append(logits.unsqueeze(1))
            recon_list.append(recon_vec.unsqueeze(1))
            
            # Collect for return_attn
            read_w_stm_list.append(r_w_stm.unsqueeze(1))
            read_w_ltm_list.append(r_w_ltm.unsqueeze(1))
            write_w_list.append(write_w.unsqueeze(1))
            adj_stm_list.append(self.memory.stm_adjacency.unsqueeze(1))
            adj_ltm_list.append(self.memory.ltm_adjacency.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)
        recon = torch.cat(recon_list, dim=1)
        
        if return_attn:
            return (
                logits,
                recon,
                (torch.cat(read_w_stm_list, dim=1), torch.cat(read_w_ltm_list, dim=1)),
                torch.cat(write_w_list, dim=1),
                (torch.cat(adj_stm_list, dim=1), torch.cat(adj_ltm_list, dim=1))
            )

        return logits, recon, None
