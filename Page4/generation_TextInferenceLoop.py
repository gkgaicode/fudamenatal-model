import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

@torch.no_grad()
def generate_text(model, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    
    # Load tokenizer to parse prompt text
    tokenizer = Tokenizer.from_file("my_tokenizer.json")
    encoded_prompt = tokenizer.encode(prompt).ids
    
    # Initialize index tensor with prompt token IDs
    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device) # Shape: [1, seq_len]
    
    for _ in range(max_new_tokens):
        # Truncate prompt context if it exceeds the maximum context length of the model
        idx_cond = idx[:, -model.max_seq_len:]
        
        # Forward pass to get raw prediction logits
        logits = model(idx_cond)
        
        # Extract the predictions for the final token position in the sequence
        logits = logits[:, -1, :] / temperature
        
        # Optional: Apply Top-K filtering to keep outputs coherent
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
        # Convert logits into a soft probability distribution
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution to select the next token ID
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token back to the context sequence loop
        idx = torch.cat((idx, next_token), dim=1)
        
        # Break generation early if the model produces an End Of Stream token
        if next_token.item() == tokenizer.token_to_id("[EOS]"):
            break
            
    # Decode the final array of token IDs back into standard readable text string
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)

# Example Execution Context:
# output_text = generate_text(trained_model, prompt="Foundational AI models are", max_new_tokens=30)
# print(output_text)
