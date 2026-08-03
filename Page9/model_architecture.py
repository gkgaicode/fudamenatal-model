import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        return torch.cat((-x[..., self.dim // 2:], x[..., :self.dim // 2]), dim=-1)

    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, x, cos, sin):
        return (x * cos) + (self._rotate_half(x) * sin)

class RoPEFlashAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(dim=self.head_dim)
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(q, T)
        cos, sin = cos.unsqueeze(0).unsqueeze(1), sin.unsqueeze(0).unsqueeze(1)
        q, k = self.rope.apply_rope(q, cos, sin), self.rope.apply_rope(k, cos, sin)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class CheckpointedBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = RoPEFlashAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        def _forward_block(h):
            return h + self.mlp(self.ln_2(h + self.attn(self.ln_1(h))))
        return checkpoint(_forward_block, x, use_reentrant=False)

class ProductionFoundationModel(nn.Module):
    def __init__(self, vocab_size=32000, d_model=2048, n_heads=16, n_layers=24, max_seq_len=4096):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[CheckpointedBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        return self.lm_head(self.ln_f(self.blocks(self.token_embedding(idx))))
