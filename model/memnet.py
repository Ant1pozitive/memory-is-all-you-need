import torch
from torch import nn
from .memory_bank import MultiHeadMemoryBank
from .controller import Controller

class MemNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.memory = MultiHeadMemoryBank(cfg.memory.slots, cfg.memory.dim, cfg.memory.heads, cfg.memory.topk, cfg.memory.policy)
        self.controller = Controller(cfg.model.vocab_size, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.memory.dim, cfg.memory.heads)

    def forward_sequence(self, input_seq: torch.Tensor, return_attn: bool = False):
        B, T = input_seq.shape
        device = input_seq.device
        memory = self.memory.reset_memory(B, device)
        h = torch.zeros(B, self.cfg.model.hidden_dim, device=device)

        logits_seq = []
        read_weights_hist = []
        write_weights_hist = []

        # Collect all keys for batch processing (partial vectorization)
        read_keys = torch.zeros(B, T, self.cfg.memory.heads, self.cfg.memory.dim, device=device)
        write_keys = torch.zeros_like(read_keys)
        write_vals = torch.zeros_like(read_keys)
        erases = torch.zeros(B, T, self.cfg.memory.heads, device=device)
        add_gates = torch.zeros_like(erases)
        read_vec = torch.zeros(B, self.cfg.memory.dim, device=device)

        for t in range(T):  # still loop due to recurrence
            x_t = input_seq[:, t]
            logits, h, read_key, write_key, write_val, erase, add_gate = self.controller(x_t, h, read_vec)
            read_keys[:, t] = read_key
            write_keys[:, t] = write_key
            write_vals[:, t] = write_val
            erases[:, t] = erase
            add_gates[:, t] = add_gate
            # read/write at each step
            beta_r = F.softplus(self.controller.beta_read)
            read_vec, read_w = self.memory.read(memory, read_key, beta_r)
            beta_w = F.softplus(self.controller.beta_write)
            memory, write_w = self.memory.write(memory, write_key, write_val, erase, add_gate, beta_w)
            logits_seq.append(logits.unsqueeze(1))
            if return_attn:
                read_weights_hist.append(read_w)
                write_weights_hist.append(write_w)

        logits_seq = torch.cat(logits_seq, dim=1)
        if return_attn:
            read_weights = torch.stack(read_weights_hist, dim=1)
            write_weights = torch.stack(write_weights_hist, dim=1)
            return logits_seq, read_weights, write_weights
        return logits_seq
