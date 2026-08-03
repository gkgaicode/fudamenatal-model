To build a foundational model completely from scratch, you start by defining a Transformer architecture. Below is a complete, minimal implementation of a Decoder-only Transformer (similar to GPT) built using PyTorch, followed by a basic pre-training loop.
## 1. Define the Transformer Architecture

import torchimport torch.nn as nnimport torch.nn.functional as F
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Key, Query, Value projections combined into one layer
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size() # Batch, Sequence Length, Channels (d_model)
        
        # Calculate Q, K, V for all heads in batch
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # Causal attention mask to prevent looking into the future
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Pre-LN architecture with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class MiniLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Blocks
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        
        # Final Language Model Head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Forward pass through layers
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

## 2. Execute the Pre-training Loop
This snippet sets up self-supervised learning, where the model predicts the next token in a sequence. [1] 

# Hyperparameters (Scaled down for demonstration)vocab_size = 50257     # Standard GPT-2 vocabulary sized_model = 256          # Embedding dimensionn_heads = 4            # Multi-head attention headsn_layers = 4           # Transformer layersmax_seq_len = 128      # Context window sizebatch_size = 32learning_rate = 3e-4
# Initialize model and optimizerdevice = 'cuda' if torch.cuda.is_available() else 'cpu'model = MiniLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Dummy dataset generation (Replace with tokenized web crawl data)# Inputs (X) shape: [batch_size, max_seq_len]# Targets (Y) shape: [batch_size, max_seq_len] (shifted by 1 token)X = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device)Y = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device)
# Simple training step
model.train()
optimizer.zero_grad()
# Forward passlogits = model(X)
# Reshape logits and targets for CrossEntropyLoss calculation# Logits: [Batch * Sequence, Vocab], Targets: [Batch * Sequence]loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
# Backward pass and update weights
loss.backward()
optimizer.step()

print(f"Training Loss: {loss.item():.4f}")

## Scale Considerations
To scale this from a toy script to a true foundational model, you would need to:

   1. Replace the dummy dataset with multi-terabyte tokenizers (like Hugging Face tokenizers).
   2. Wrap the model in PyTorch FSDP (Fully Sharded Data Parallel) or DeepSpeed to split layers across thousands of cluster nodes.

If you are ready to take next steps, let me know:

* Do you want to see how to implement distributed training (FSDP/DeepSpeed) code?
* Would you like code for tokenizing raw text files before feeding them to this model?
* Are you looking for code to generate text (inference loop) from this model?

