# Hyperparameters (Scaled down for demonstration)
vocab_size = 50257     # Standard GPT-2 vocabulary size
d_model = 256          # Embedding dimension
n_heads = 4            # Multi-head attention heads
n_layers = 4           # Transformer layers
max_seq_len = 128      # Context window size
batch_size = 32
learning_rate = 3e-4

# Initialize model and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Dummy dataset generation (Replace with tokenized web crawl data)
# Inputs (X) shape: [batch_size, max_seq_len]
# Targets (Y) shape: [batch_size, max_seq_len] (shifted by 1 token)
X = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device)
Y = torch.randint(0, vocab_size, (batch_size, max_seq_len)).to(device)

# Simple training step
model.train()
optimizer.zero_grad()

# Forward pass
logits = model(X)

# Reshape logits and targets for CrossEntropyLoss calculation
# Logits: [Batch * Sequence, Vocab], Targets: [Batch * Sequence]
loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))

# Backward pass and update weights
loss.backward()
optimizer.step()

print(f"Training Loss: {loss.item():.4f}")
