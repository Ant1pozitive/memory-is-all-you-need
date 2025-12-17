"""
Controller network that generates multi-head keys/values and outputs.
"""
import torch
from torch import nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, mem_dim: int, n_heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # controller input: embedding + read vector (mem_dim)
        self.lstm = nn.LSTMCell(embed_dim + mem_dim, hidden_dim)
        self.hidden_to_read_key = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_write_key = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_write_val = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_erase = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.output_head = nn.Linear(hidden_dim + mem_dim, vocab_size)
        # temperature params
        self.beta_read = nn.Parameter(torch.tensor(5.0))
        self.beta_write = nn.Parameter(torch.tensor(5.0))
        self.n_heads = n_heads
        self.mem_dim = mem_dim

    def forward(self, x_t, h, c, read_vec):
        # x_t: (B,)
        emb = self.embed(x_t) # (B,E)
        inp = torch.cat([emb, read_vec], dim=-1)
        h_new, c_new = self.lstm(inp, (h, c))
        # produce keys/values
        read_key = self.hidden_to_read_key(h_new).view(-1, self.n_heads, self.mem_dim)
        write_key = self.hidden_to_write_key(h_new).view(-1, self.n_heads, self.mem_dim)
        write_val = torch.tanh(self.hidden_to_write_val(h_new)).view(-1, self.n_heads, self.mem_dim)
        erase = torch.sigmoid(self.hidden_to_erase(h_new)).view(-1, self.n_heads, self.mem_dim)
        logits = self.output_head(torch.cat([h_new, read_vec], dim=-1))
        return logits, h_new, c_new, read_key, write_key, write_val, erase
