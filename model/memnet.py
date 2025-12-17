"""
Memory-augmented network wrapper that ties controller + memory bank.
Forward over a sequence: returns logits per timestep. Optionally returns read/write weights history for analysis.
"""
import torch
from torch import nn
from .memory_bank import MultiHeadMemoryBank
from .controller import Controller

class MemNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.memory = MultiHeadMemoryBank(cfg.mem_slots, cfg.mem_dim, n_heads=cfg.mem_heads, topk=cfg.topk)
        self.controller = Controller(cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.mem_dim, cfg.mem_heads)

    def forward_sequence(self, input_seq: torch.Tensor, return_attn: bool = False):
        # input_seq: (B, T)
        B, T = input_seq.shape
        device = input_seq.device
        memory = self.memory.reset_memory(B, device)
        h = torch.zeros(B, self.cfg.hidden_dim, device=device)
        c = torch.zeros(B, self.cfg.hidden_dim, device=device)
        read_vec = torch.zeros(B, self.cfg.mem_dim, device=device)

        logits_seq = []
        read_weights_hist = []
        write_weights_hist = []

        for t in range(T):
            x_t = input_seq[:, t]
            logits, h, c, read_key, write_key, write_val, erase = self.controller(x_t, h, c, read_vec)
            # read
            read_vec, read_w = self.memory.read(memory, read_key, beta=float(self.controller.beta_read.abs().item() + 1e-6))
            # write
            memory, write_w = self.memory.write(memory, write_key, write_val, erase=erase, beta=float(self.controller.beta_write.abs().item() + 1e-6))

            logits_seq.append(logits.unsqueeze(1))
            if return_attn:
                read_weights_hist.append(read_w.detach().cpu())
                write_weights_hist.append(write_w.detach().cpu())

        logits_seq = torch.cat(logits_seq, dim=1)
        if return_attn:
            # stack along time
            read_weights = torch.stack(read_weights_hist, dim=1) # (B, T, H, N)
            write_weights = torch.stack(write_weights_hist, dim=1)
            return logits_seq, read_weights, write_weights
        return logits_seq
