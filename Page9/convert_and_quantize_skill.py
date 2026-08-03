import os
import torch
from transformers import PretrainedConfig
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

class CustomHFModelConfig(PretrainedConfig):
    model_type = "custom_rope_transformer"
    def __init__(self, vocab_size=32000, d_model=2048, n_heads=16, n_layers=24, max_seq_len=4096, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

def run(ds_checkpoint_folder, hf_out_dir, awq_out_dir):
    # 1. Consolidate sharded states to clean HF layout
    os.makedirs(hf_out_dir, exist_ok=True)
    consolidated_path = os.path.join(ds_checkpoint_folder, "consolidated_weights.pt")
    
    raw_state_dict = torch.load(consolidated_path, map_location="cpu")
    clean_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
    
    config = CustomHFModelConfig()
    config.save_pretrained(hf_out_dir)
    torch.save(clean_state_dict, os.path.join(hf_out_dir, "pytorch_model.bin"))
    print("[SKILL] Flat 16-bit Hugging Face base directory compiled successfully.")

    # 2. Run 4-Bit Activation Aware Quantization
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(hf_out_dir, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_out_dir, trust_remote_code=True)
    
    # Calibration proxy text sequence array blocks
    calib_samples = ["Sample sentence validating channel activation distributions."]
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_samples)
    
    model.save_quantized(awq_out_dir)
    tokenizer.save_pretrained(awq_out_dir)
    print(f"[SKILL COMPLETED] Compressed 4-bit weights deployed safely to: {awq_out_dir}")
