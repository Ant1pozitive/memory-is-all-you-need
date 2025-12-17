import torch
from torch import nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, mem_dim: int, n_heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRUCell(embed_dim + mem_dim, hidden_dim)
        self.hidden_to_read_key = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_write_key = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_write_val = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_erase = nn.Linear(hidden_dim, n_heads)  # scalar per head
        self.hidden_to_add = nn.Linear(hidden_dim, n_heads)  # add gate scalar
        self.output_head = nn.Linear(hidden_dim + mem_dim, vocab_size)
        self.beta_read = nn.Parameter(torch.ones(n_heads))  # per-head
        self.beta_write = nn.Parameter(torch.ones(n_heads))
        self.n_heads = n_heads
        self.mem_dim = mem_dim
        self.head_merge = nn.Linear(n_heads * mem_dim, mem_dim)

    def forward(self, x_t, h, read_vec):
        emb = self.embed(x_t)
        inp = torch.cat([emb, read_vec], dim=-1)
        h_new = self.gru(inp, h)
        read_key = self.hidden_to_read_key(h_new).view(-1, self.n_heads, self.mem_dim)
        write_key = self.hidden_to_write_key(h_new).view(-1, self.n_heads, self.mem_dim)
        write_val = self.hidden_to_write_val(h_new).view(-1, self.n_heads, self.mem_dim)
        erase = torch.sigmoid(self.hidden_to_erase(h_new))  # (B,H)
        add_gate = torch.sigmoid(self.hidden_to_add(h_new))  # (B,H)
        logits = self.output_head(torch.cat([h_new, read_vec], dim=-1))
        return logits, h_new, read_key, write_key, write_val, erase, add_gate
