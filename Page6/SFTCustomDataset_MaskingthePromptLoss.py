from torch.utils.data import Dataset
from tokenizers import Tokenizer

class SupervisedFineTuningDataset(Dataset):
    def __init__(self, data_pairs, tokenizer_path="my_tokenizer.json", max_len=2048):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.encoded_pairs = []
        
        # Pull standard internal token boundary indicators
        pad_id = self.tokenizer.token_to_id("[PAD]")
        eos_id = self.tokenizer.token_to_id("[EOS]")

        for prompt_str, response_str in data_pairs:
            # Tokenize system string inputs separately
            p_tokens = self.tokenizer.encode(prompt_str).ids
            r_tokens = self.tokenizer.encode(response_str).ids + [eos_id]
            
            # Combine arrays
            input_ids = p_tokens + r_tokens
            # Label mask setup: Use -100 to completely ignore the prompt tokens during loss calculation
            target_ids = [-100] * len(p_tokens) + r_tokens
            
            # Apply padding or truncation down to max_len bounds
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                target_ids = target_ids[:max_len]
            else:
                padding_length = max_len - len(input_ids)
                input_ids += [pad_id] * padding_length
                target_ids += [-100] * padding_length # Do not compute loss on padding
                
            self.encoded_pairs.append((torch.tensor(input_ids), torch.tensor(target_ids)))

    def __len__(self):
        return len(self.encoded_pairs)

    def __getitem__(self, idx):
        return self.encoded_pairs[idx]
