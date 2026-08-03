import math
import torch
import deepspeed
from torch.utils.data import Dataset, DataLoader
# Assumes 'AdvancedFoundationModel' is defined or imported here

class PretrainingDataset(Dataset):
    def __init__(self, seq_len=2048):
        self.seq_len = seq_len
        # Using mock token arrays for demonstration
        self.data = torch.randint(0, 32000, (50000,))

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len], self.data[idx + 1 : idx + self.seq_len + 1]

@torch.no_grad()
def evaluate_model_perplexity(model_engine, val_dataloader):
    """Computes exact validation perplexity over evaluation batches."""
    model_engine.eval()
    total_loss = 0.0
    total_steps = 0
    
    for x, y in val_dataloader:
        x, y = x.to(model_engine.device), y.to(model_engine.device)
        logits = model_engine(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        total_loss += loss.item()
        total_steps += 1
        if total_steps >= 20: # Caps eval loop runtime length
            break
            
    avg_loss = total_loss / max(total_steps, 1)
    # Perplexity mathematically maps to the exponent of cross entropy loss
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    model_engine.train()
    return avg_loss, perplexity

def run_main_pretraining():
    # 1. Initialize core system components
    deepspeed.init_distributed(dist_backend="nccl")
    
    model = AdvancedFoundationModel()
    train_dataset = PretrainingDataset()
    val_dataset = PretrainingDataset()
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 2. Bind model, configurations, and optimizer to DeepSpeed engine wrap
    # Optimizers are handled automatically based on parameters provided to initialize API
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config_zero3.json"
    )
    
    # 3. Main Pre-training Loop
    for epoch in range(1):
        for step, (x, y) in enumerate(train_loader):
            x = x.to(model_engine.device)
            y = y.to(model_engine.device)
            
            # Forward pass
            logits = model_engine(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # DeepSpeed backpropagation engine steps
            model_engine.backward(loss)
            model_engine.step()
            
            # Continuous quality monitoring output block
            if step % 10 == 0 and deepspeed.comm.get_rank() == 0:
                print(f"Epoch {epoch} | Step {step} | Train Loss: {loss.item():.4f}")
                
            # Periodically calculate out-of-sample evaluation metrics
            if step % 100 == 0:
                val_loss, val_ppl = evaluate_model_perplexity(model_engine, val_loader)
                if deepspeed.comm.get_rank() == 0:
                    print(f"\n--- EVALUATION PERIODIC METRICS ---")
                    print(f"Validation Target Loss: {val_loss:.4f}")
                    print(f"Validation Target Perplexity: {val_ppl:.4f}\n----------------------------------")

if __name__ == "__main__":
    run_main_pretraining()
