import os
import json
import torch
from transformers import PretrainedConfig, PreTrainedModel
# Imports your model framework class definition setup
# from custom_model import SFTProductionModel

class CustomHFModelConfig(PretrainedConfig):
    model_type = "custom_rope_transformer"
    def __init__(self, vocab_size=32000, d_model=2048, n_heads=16, n_layers=24, max_seq_len=4096, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

def export_and_convert_weights(deepspeed_checkpoint_folder, output_hf_dir):
    """
    Consolidates DeepSpeed ZeRO sharded partition files 
    and saves them as a single clean Hugging Face model repository.
    """
    os.makedirs(output_hf_dir, exist_ok=True)
    
    # 1. Consolidate sharded weights into a clean, flat PyTorch state dict.
    # DeepSpeed includes a utility tool called 'zero_to_fp32.py' inside its repository source base
    # Run: python zero_to_fp32.py deepspeed_checkpoint_folder consolidated_weights.pt
    # Here, we assume you have the flat consolidated_weights.pt ready:
    consolidated_path = os.path.join(deepspeed_checkpoint_folder, "consolidated_weights.pt")
    if not os.path.exists(consolidated_path):
        raise FileNotFoundError(f"Please use zero_to_fp32.py to generate {consolidated_path} first.")

    custom_state_dict = torch.load(consolidated_path, map_dict="cpu")
    hf_state_dict = {}

    # 2. Translate your internal model parameter keys into clean, standard keys
    # Example state translation block:
    for key, tensor in custom_state_dict.items():
        # Strip DeepSpeed engine module wrappers if present
        clean_key = key.replace("module.", "")
        
        # Translation maps can be inserted here if tensor names diverge from defaults
        # For our SFTProductionModel, names match cleanly, so we can copy directly:
        hf_state_dict[clean_key] = tensor

    # 3. Create, write, and save standard Hugging Face model properties files
    config = CustomHFModelConfig()
    config.save_pretrained(output_hf_dir)
    
    # Save target parameters binary index model weights tracking file
    torch.save(hf_state_dict, os.path.join(output_hf_dir, "pytorch_model.bin"))
    
    # Copy your tokenizer file into the deployment directory
    if os.path.exists("my_tokenizer.json"):
        import shutil
        shutil.copy("my_tokenizer.json", os.path.join(output_hf_dir, "tokenizer.json"))
        
    print(f"Success! Model weights exported to Hugging Face hub layout at: {output_hf_dir}")

if __name__ == "__main__":
    # Example workflow trigger call block:
    # export_and_convert_weights("./checkpoints/global_step100", "./my_hf_aligned_model")
    pass
