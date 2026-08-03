import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Fused projection layer
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate queries, keys, values and reshape for multi-head layout
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # PyTorch Native SDPA automatically dispatches to FlashAttention-2 
        # when running on modern GPUs with half-precision (FP16/BF16).
        # is_causal=True automatically constructs and applies the look-ahead mask.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        
        # Re-assemble back to flat channel layout
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class FlashTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = FlashCausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class AdvancedFoundationModel(nn.Module):
    def __init__(self, vocab_size=32000, d_model=2048, n_heads=16, n_layers=24, max_seq_len=2048):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(*[FlashTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)
