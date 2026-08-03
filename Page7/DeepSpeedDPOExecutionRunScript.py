# Run with: deepspeed --num_gpus=NUM_GPUS dpo_train.py
import copy
from torch.utils.data import DataLoader
# Assumes SFTProductionModel & functions above are active in runtime environment

def train_dpo_alignment():
    deepspeed.init_distributed(dist_backend="nccl")
    
    # 1. Initialize two copies: The active Policy model and a frozen Reference model copy
    policy_model = SFTProductionModel(max_seq_len=2048)
    reference_model = copy.deepcopy(policy_model) # Keep frozen to prevent baseline drift
    reference_model.eval()

    # 2. Package datasets with preference feedback loops
    preference_dataset = [
        ("User: Tell me a joke.\nAssistant:", "Why don't scientists trust atoms? Because they make up everything!", "I am a language model and cannot tell jokes."),
    ]
    collator = DPODataCollator()
    dataloader = DataLoader(preference_dataset, batch_size=2, collate_fn=collator)

    # Wrap policy model in DeepSpeed engine execution environment framework loop
    policy_engine, _, _, _ = deepspeed.initialize(model=policy_model, config="ds_config_zero3.json")
    reference_device = policy_engine.device
    reference_model.to(reference_device)

    policy_engine.train()
    for c_ids, c_labels, r_ids, r_labels in dataloader:
        c_ids, c_labels = c_ids.to(policy_engine.device), c_labels.to(policy_engine.device)
        r_ids, r_labels = r_ids.to(policy_engine.device), r_labels.to(policy_engine.device)

        # Step A: Run forward passes on the active policy model
        policy_c_logits = policy_engine(c_ids)
        policy_r_logits = policy_engine(r_ids)
        
        policy_c_logps = compute_logps(policy_c_logits, c_labels)
        policy_r_logps = compute_logps(policy_r_logits, r_labels)

        # Step B: Run forward passes on the frozen reference model
        with torch.no_grad():
            ref_c_logits = reference_model(c_ids)
            ref_r_logits = reference_model(r_ids)
            
            ref_c_logps = compute_logps(ref_c_logits, c_labels)
            ref_r_logps = compute_logps(ref_r_logits, r_labels)

        # Step C: Compute DPO cross-entropy contrast ratios
        loss, chosen_r, rejected_r = dpo_loss(policy_c_logps, policy_r_logps, ref_c_logps, ref_r_logps)

        policy_engine.backward(loss)
        policy_engine.step()

        if deepspeed.comm.get_rank() == 0:
            print(f"DPO Loss: {loss.item():.4f} | Chosen Reward: {chosen_r.item():.4f} | Rejected Margin: {(chosen_r - rejected_r).item():.4f}")

if __name__ == "__main__":
    import os
    if "WORLD_SIZE" in os.environ: train_dpo_alignment()
