import torch
from torch import nn
import torch.nn.functional as F

class BaseController(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, mem_dim: int, n_heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_to_read_key = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_write_key = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_write_val = nn.Linear(hidden_dim, n_heads * mem_dim)
        self.hidden_to_erase = nn.Linear(hidden_dim, n_heads)
        self.hidden_to_add = nn.Linear(hidden_dim, n_heads)
        self.output_head = nn.Linear(hidden_dim + mem_dim, vocab_size)
        self.beta_read = nn.Parameter(torch.ones(n_heads))
        self.beta_write = nn.Parameter(torch.ones(n_heads))
        self.n_heads = n_heads
        self.mem_dim = mem_dim

class GRUController(BaseController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim, mem_dim, hidden_dim = args[1], args[3], args[2]
        self.gru = nn.GRUCell(embed_dim + mem_dim, hidden_dim)

    def forward(self, x_t, h, read_vec):
        emb = self.embed(x_t)
        inp = torch.cat([emb, read_vec], dim=-1)
        h_new = self.gru(inp, h)
        return self._common_forward(h_new, read_vec)

    def _common_forward(self, h_new, read_vec):
        read_key = self.hidden_to_read_key(h_new).view(-1, self.n_heads, self.mem_dim)
        write_key = self.hidden_to_write_key(h_new).view(-1, self.n_heads, self.mem_dim)
        write_val = self.hidden_to_write_val(h_new).view(-1, self.n_heads, self.mem_dim)
        erase = torch.sigmoid(self.hidden_to_erase(h_new))
        add_gate = torch.sigmoid(self.hidden_to_add(h_new))
        logits = self.output_head(torch.cat([h_new, read_vec], dim=-1))
        return logits, h_new, read_key, write_key, write_val, erase, add_gate

class TransformerController(BaseController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim, mem_dim, hidden_dim = args[1], args[3], args[2]
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim + mem_dim, nhead=4, dim_feedforward=hidden_dim)

    def forward(self, x_t, h, read_vec):
        emb = self.embed(x_t).unsqueeze(0)  # seq dim
        inp = torch.cat([emb, read_vec.unsqueeze(0)], dim=-1)
        h_new = self.transformer(inp).squeeze(0)
        return self._common_forward(h_new, read_vec)
