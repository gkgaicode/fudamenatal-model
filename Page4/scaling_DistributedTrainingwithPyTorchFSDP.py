import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# A memory-mapped dataset that reads chunks directly from your binary file
class BinDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = torch.from_numpy(torch.from_file(bin_path, dtype=torch.int16).numpy().astype('int64'))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # Sliding window sequence extraction
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1] # Next-token prediction target
        return x, y

def setup_distributed():
    # Setup process groups for cross-GPU communication
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    dist.destroy_process_group()

def run_fsdp_training():
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 1. Instantiate your model (Using MiniLanguageModel from previous example)
    # Ensure MiniLanguageModel code is imported or pasted above this
    raw_model = MiniLanguageModel(vocab_size=32000, d_model=1024, n_heads=16, n_layers=12, max_seq_len=1024)
    raw_model.to(local_rank)
    
    # 2. Wrap model in FSDP to break it apart and shard it across active GPUs
    model = FSDP(raw_model)
    
    # 3. Setup optimizer *after* wrapping with FSDP
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # 4. Prepare data distribution across GPUs
    dataset = BinDataset("pretraining_dataset.bin", seq_len=1024)
    sampler = DistributedSampler(dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size())
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    model.train()
    for epoch in range(1):
        sampler.set_epoch(epoch)
        for x, y in dataloader:
            x, y = x.to(local_rank), y.to(local_rank)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            if dist.get_rank() == 0:
                print(f"Distributed Step Complete. Loss: {loss.item():.4f}")
                
    cleanup_distributed()

if __name__ == "__main__":
    # Standard safeguard check before initiating distributed threads
    if "WORLD_SIZE" in os.environ:
        run_fsdp_training()
