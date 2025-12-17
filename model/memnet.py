import torch
from torch import nn
from .memory_bank import MultiHeadMemoryBank
from .controller import GRUController, TransformerController

class MemNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.memory = MultiHeadMemoryBank(cfg.memory.slots, cfg.memory.dim, cfg.memory.heads, cfg.memory.topk, cfg.memory.policy)
        controller_cls = GRUController if cfg.model.controller_type == "gru" else TransformerController
        self.controller = controller_cls(cfg.model.vocab_size, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.memory.dim, cfg.memory.heads)

    def forward_sequence(self, input_seq: torch.Tensor, return_attn: bool = False):
        B, T = input_seq.shape
        device = input_seq.device
        memory = self.memory.reset_memory(B, device)
        h = torch.zeros(B, self.cfg.model.hidden_dim, device=device)

        logits_seq = []
        read_weights_hist = []
        write_weights_hist = []
        memory_hist = [] if return_attn else None

        for t in range(T):
            if return_attn:
                memory_hist.append(memory.clone().detach().cpu())
            x_t = input_seq[:, t]
            logits, h, read_key, write_key, write_val, erase, add_gate = self.controller(x_t, h, read_vec if 'read_vec' in locals() else torch.zeros(B, self.cfg.memory.dim, device=device))
            beta_r = F.softplus(self.controller.beta_read).clamp(1, 20)
            read_vec, read_w = self.memory.read(memory, read_key, beta_r)
            beta_w = F.softplus(self.controller.beta_write).clamp(1, 20)
            memory, write_w = self.memory.write(memory, write_key, write_val, erase, add_gate, beta_w)
            logits_seq.append(logits.unsqueeze(1))
            if return_attn:
                read_weights_hist.append(read_w.detach().cpu())
                write_weights_hist.append(write_w.detach().cpu())

        logits_seq = torch.cat(logits_seq, dim=1)
        if return_attn:
            read_weights = torch.stack(read_weights_hist, dim=1)
            write_weights = torch.stack(write_weights_hist, dim=1)
            return logits_seq, read_weights, write_weights, torch.stack(memory_hist, dim=0)
        return logits_seq
