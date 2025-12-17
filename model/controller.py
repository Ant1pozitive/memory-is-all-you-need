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
        embed_dim = args[1]
        self.pos_emb = nn.Parameter(torch.randn(1, 512, embed_dim))  # max seq
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim * 2, nhead=8, dim_feedforward=512),
            num_layers=cfg.model.num_layers
        )

    def forward(self, input_seq, read_vec):
        # input_seq: (B, T), read_vec: (B, D)
        emb = self.embed(input_seq)  # (B, T, D)
        read_token = read_vec.unsqueeze(1)  # (B, 1, D)
        inp = torch.cat([emb, read_token], dim=1)  # prepend or append memory token
        pos = self.pos_emb[:, :inp.shape[1]]
        inp = inp + pos
        out = self.transformer(inp.transpose(0,1)).transpose(0,1)  # (B, T+1, D)
        controller_out = out[:, -1]  # last token or mean
        return self._common_forward(controller_out, read_vec)
