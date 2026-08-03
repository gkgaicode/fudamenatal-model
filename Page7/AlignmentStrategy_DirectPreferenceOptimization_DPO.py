import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

class DPODataCollator:
    """
    Prepares preference pairs. For a single prompt, it packages both the 
    chosen (good) and rejected (bad) completions into one combined batch tensor.
    """
    def __init__(self, tokenizer_path="my_tokenizer.json", max_len=2048):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")

    def __call__(self, batch_triplets):
        # batch_triplets format: [ (prompt, chosen, rejected), ... ]
        chosen_input_ids, chosen_labels = [], []
        rejected_input_ids, rejected_labels = [], []

        for prompt, chosen, rejected in batch_triplets:
            p_tokens = self.tokenizer.encode(prompt).ids
            c_tokens = self.tokenizer.encode(chosen).ids + [self.eos_id]
            r_tokens = self.tokenizer.encode(rejected).ids + [self.eos_id]

            # Process Chosen sequence
            c_input = (p_tokens + c_tokens)[:self.max_len]
            c_label = ([-100] * len(p_tokens) + c_tokens)[:self.max_len]
            c_input += [self.pad_id] * (self.max_len - len(c_input))
            c_label += [-100] * (self.max_len - len(c_label))
            
            # Process Rejected sequence
            r_input = (p_tokens + r_tokens)[:self.max_len]
            r_label = ([-100] * len(p_tokens) + r_tokens)[:self.max_len]
            r_input += [self.pad_id] * (self.max_len - len(r_input))
            r_label += [-100] * (self.max_len - len(r_label))

            chosen_input_ids.append(torch.tensor(c_input))
            chosen_labels.append(torch.tensor(c_label))
            rejected_input_ids.append(torch.tensor(r_input))
            rejected_labels.append(torch.tensor(r_label))

        return (
            torch.stack(chosen_input_ids), torch.stack(chosen_labels),
            torch.stack(rejected_input_ids), torch.stack(rejected_labels)
        )

def compute_logps(logits, labels):
    """Extracts log-probabilities only for non-masked (-100) target tokens."""
    # Shift logits and labels by 1 position for causal prediction next-token tracking
    per_token_logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_labels = labels[:, 1:].clone()
    
    loss_mask = target_labels != -100
    target_labels[target_labels == -100] = 0 # Dummy fill to prevent index crash
    
    # Gather the log-probabilities corresponding to the actual correct tokens
    gather_logps = torch.gather(per_token_logps, dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
    return (gather_logps * loss_mask).sum(dim=-1)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Mathematical implementation of the implicit DPO preference loss equation."""
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps
    
    logits = policy_log_ratios - ref_log_ratios
    losses = -F.logsigmoid(beta * logits)
    
    # Calculate rewards for performance metric tracking logs
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()
