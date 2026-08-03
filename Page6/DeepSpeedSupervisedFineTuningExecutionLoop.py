import deepspeed
from torch.utils.data import DataLoader

def execute_sft_pipeline():
    deepspeed.init_distributed(dist_backend="nccl")
    
    # 1. Provide downstream task instruction samples
    sample_instruction_data = [
        ("User: Write code to parse json in python.\nAssistant:", "Here is the code:\nimport json..."),
        ("User: What is the capital of France?\nAssistant:", "The capital of France is Paris.")
    ]
    
    # Initialize the custom masked dataset
    sft_dataset = SupervisedFineTuningDataset(sample_instruction_data, max_len=2048)
    train_loader = DataLoader(sft_dataset, batch_size=4, shuffle=True)
    
    # 2. Instantiate our advanced RoPE + Activation Checkpointed model
    model = SFTProductionModel(max_seq_len=2048)
    
    # Initialize DeepSpeed framework hook wraps
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config_zero3.json"
    )
    
    # 3. Instruction Training Loop Execution
    model_engine.train()
    for epoch in range(3): # SFT typically runs for fewer epochs (e.g., 3-5) compared to pre-training
        for step, (x, y) in enumerate(train_loader):
            x = x.to(model_engine.device)
            y = y.to(model_engine.device)
            
            logits = model_engine(x)
            
            # PyTorch cross_entropy safely skips index entries carrying target value matching -100
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1), 
                ignore_index=-100
            )
            
            model_engine.backward(loss)
            model_engine.step()
            
            if deepspeed.comm.get_rank() == 0:
                print(f"SFT Iteration -> Epoch: {epoch} | Step: {step} | Target Instruction Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # Safety wrapper block checking environment framework properties
    import os
    if "WORLD_SIZE" in os.environ:
        execute_sft_pipeline()
