import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np

# 1. Train the tokenizer on raw text data
def train_my_tokenizer(text_files_list, vocab_size=32000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )
    
    tokenizer.train(text_files_list, trainer)
    tokenizer.save("my_tokenizer.json")
    return tokenizer

# 2. Tokenize text documents and save as binary files for fast memory-mapping
def process_and_save_data(text_files_list, tokenizer_path="my_tokenizer.json"):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    all_tokens = []
    
    for file_path in text_files_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Convert text to token IDs
        encoded = tokenizer.encode(text)
        all_tokens.extend(encoded.ids)
    
    # Store token IDs as a high-performance numpy array of 16-bit integers
    token_arr = np.array(all_tokens, dtype=np.uint16)
    token_arr.tofile("pretraining_dataset.bin")
    print(f"Saved {len(token_arr)} tokens to binary file.")

# Quick setup block
if __name__ == "__main__":
    # Create sample dummy files to demonstrate the flow
    with open("sample.txt", "w") as f: f.write("Foundational models learn text patterns from huge corpuses.")
    
    train_my_tokenizer(["sample.txt"], vocab_size=1000)
    process_and_save_data(["sample.txt"])
