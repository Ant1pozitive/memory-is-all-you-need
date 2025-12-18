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
            cfg.memory.slots, cfg.memory.dim, cfg.memory.heads, cfg.memory.topk,
            cfg.memory.policy, cfg.memory.use_decay_gate, cfg.memory.decay_rate,
            cfg.memory.bottleneck_dim
        )
        self.controller = TransformerController(
            cfg.model.vocab_size, cfg.model.embed_dim, cfg.model.hidden_dim,
            cfg.memory.dim, cfg.memory.heads, cfg.model.num_layers,
            cfg.model.num_heads_attn, cfg.model.max_seq_len
        )

    def forward(self, input_seq: torch.Tensor, return_attn: bool = False):
        B, T = input_seq.shape
        device = input_seq.device
        memory = self.memory.reset_memory(B, device)
        read_vec = torch.zeros(B, self.cfg.memory.dim, device=device)

        logits_list = []
        read_weights_hist = [] if return_attn else None
        write_weights_hist = [] if return_attn else None

        for t in range(T):
            current_input = input_seq[:, :t+1]
            logits, read_key, write_key, write_val, erase, add_gate = self.controller(
                current_input, read_vec, t
            )

            beta_r = F.softplus(self.controller.beta_read).clamp(1, 20)
            new_read_vec, read_w = self.memory.read(memory, read_key, beta_r)
            
            # Residual Connection: current read influenced by previous state
            read_vec = read_vec + new_read_vec 

            beta_w = F.softplus(self.controller.beta_write).clamp(1, 20)
            memory, write_w = self.memory.write(memory, write_key, write_val, erase, add_gate, beta_w)

            logits_list.append(logits.unsqueeze(1))

            if return_attn:
                read_weights_hist.append(read_w.detach().cpu())
                write_weights_hist.append(write_w.detach().cpu())

        logits = torch.cat(logits_list, dim=1)

        if return_attn:
            return logits, torch.stack(read_weights_hist, dim=1), torch.stack(write_weights_hist, dim=1)
        return logits