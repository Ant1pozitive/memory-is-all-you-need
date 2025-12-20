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
            cfg.memory.bottleneck_dim, cfg.memory.n_synthesis_layers, cfg.memory.synthesis_heads
        )
        self.controller = TransformerController(
            cfg.model.vocab_size, cfg.model.embed_dim, cfg.model.hidden_dim,
            cfg.memory.dim, cfg.memory.heads, cfg.model.num_layers,
            cfg.model.num_heads_attn, cfg.model.max_seq_len
        )
        
        # Hallucination Head: Tries to reconstruct the original input from the memory read vector.
        # This ensures the memory captures sufficient information to "imagine" the input.
        self.hallucination_head = nn.Linear(cfg.memory.dim, cfg.model.embed_dim)

    def forward(self, input_seq: torch.Tensor, return_attn: bool = False):
        B, T = input_seq.shape
        device = input_seq.device
        memory = self.memory.reset_memory(B, device)
        read_vec = torch.zeros(B, self.cfg.memory.dim, device=device)

        logits_list = []
        recon_list = [] # For hallucination loss
        
        read_weights_hist = [] if return_attn else None
        write_weights_hist = [] if return_attn else None

        for t in range(T):
            current_input = input_seq[:, :t+1]
            logits, read_key, write_key, write_val, erase, add_gate = self.controller(
                current_input, read_vec, t
            )

            # 1. READ
            beta_r = F.softplus(self.controller.beta_read).clamp(1, 20)
            new_read_vec, read_w = self.memory.read(memory, read_key, beta_r)
            read_vec = read_vec + new_read_vec # Residual connection

            # 2. WRITE
            beta_w = F.softplus(self.controller.beta_write).clamp(1, 20)
            memory, write_w = self.memory.write(memory, write_key, write_val, erase, add_gate, beta_w)

            # 3. SYNTHESIZE (Imaginative Replay)
            # Run synthesis periodically to allow slots to share info
            if t % self.cfg.memory.synthesis_interval == 0:
                memory = self.memory.synthesize(memory)

            # 4. HALLUCINATE (Reconstruction)
            # Try to predict the *current* input embedding using only the *read* vector
            # We don't have the raw embedding here conveniently exposed from controller, 
            # so we'll return the raw reconstruction vector and compute loss in train.py against targets
            recon_vec = self.hallucination_head(read_vec)
            
            logits_list.append(logits.unsqueeze(1))
            recon_list.append(recon_vec.unsqueeze(1))

            if return_attn:
                read_weights_hist.append(read_w.detach().cpu())
                write_weights_hist.append(write_w.detach().cpu())

        logits = torch.cat(logits_list, dim=1)
        recon = torch.cat(recon_list, dim=1) # [B, T, EmbedDim]

        if return_attn:
            return logits, recon, torch.stack(read_weights_hist, dim=1), torch.stack(write_weights_hist, dim=1)
        return logits, recon