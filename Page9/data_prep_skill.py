import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def run(text_files_list, vocab_size=32000):
    # 1. Train Custom BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    tokenizer.train(text_files_list, trainer)
    tokenizer.save("my_tokenizer.json")
    
    # 2. Compile Text to Fast-Read Binary Dataset
    all_tokens = []
    for file_path in text_files_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_tokens.extend(tokenizer.encode(f.read()).ids)
            
    token_arr = np.array(all_tokens, dtype=np.uint16)
    token_arr.tofile("pretraining_dataset.bin")
    print(f"[SKILL COMPLETED] Compiled {len(token_arr)} tokens into binary memory maps.")
