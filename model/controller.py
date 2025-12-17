import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerController(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, mem_dim: int, n_heads: int, num_layers: int, num_heads_attn: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, embed_dim))
        self.mem_proj = nn.Linear(mem_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads_attn, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_read_key = nn.Linear(embed_dim, n_heads * mem_dim)
        self.to_write_key = nn.Linear(embed_dim, n_heads * mem_dim)
        self.to_write_val = nn.Linear(embed_dim, n_heads * mem_dim)
        self.to_erase = nn.Linear(embed_dim, n_heads)
        self.to_add = nn.Linear(embed_dim, n_heads)
        self.output_head = nn.Linear(embed_dim + mem_dim, vocab_size)

        self.beta_read = nn.Parameter(torch.ones(n_heads))
        self.beta_write = nn.Parameter(torch.ones(n_heads))
        self.n_heads = n_heads
        self.mem_dim = mem_dim

    def forward(self, input_seq: torch.Tensor, read_vec: torch.Tensor):
        B, T = input_seq.shape
        emb = self.embed(input_seq)  # (B, T, E)
        pos = self.pos_emb[:, :T, :].to(emb.device)
        emb = emb + pos

        mem_token = self.mem_proj(read_vec).unsqueeze(1)  # (B, 1, E)
        seq_with_mem = torch.cat([emb, mem_token], dim=1)  # (B, T+1, E)

        out = self.transformer(seq_with_mem)  # (B, T+1, E)
        controller_state = out[:, -1]  # use memory token output

        read_key = self.to_read_key(controller_state).view(B, self.n_heads, self.mem_dim)
        write_key = self.to_write_key(controller_state).view(B, self.n_heads, self.mem_dim)
        write_val = self.to_write_val(controller_state).view(B, self.n_heads, self.mem_dim)
        erase = torch.sigmoid(self.to_erase(controller_state))
        add_gate = torch.sigmoid(self.to_add(controller_state))

        logits = self.output_head(torch.cat([controller_state, read_vec], dim=-1))

        return logits, read_key, write_key, write_val, erase, add_gate
