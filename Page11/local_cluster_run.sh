#!/bin/bash

# Terminate execution paths cleanly if any individual sub-command throws a failure flag
set -e

# --- 1. Local Device Auditing ---
# Query system configurations to find out exactly how many active GPUs are inside this machine
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "=== INITIALIZING LOCAL HIGH-THROUGHPUT ENGINE ==="
echo "Detected Active On-Board Hardware GPUs: $NUM_GPUS"
echo "================================================="

# Force PyTorch allocator optimization configurations to prevent VRAM memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- 2. Execution Routing Sequence ---
echo "[STAGE 1/3] Triggering Data Preparation and Custom Tokenizer Assembly..."
python3 agent.py --stage DATA_PREP --text_files ["corpus.txt"] --vocab_size 32000

echo "[STAGE 2/3] Launching DeepSpeed Multi-GPU Training Infrastructure Loop..."
torchrun --nproc_per_node=$NUM_GPUS agent.py --stage PRETRAIN_OR_SFT --num_gpus=$NUM_GPUS

echo "[STAGE 3/3] Consolidating Weights and Executing 4-Bit AWQ Quantization..."
python3 agent.py --stage EXPORT_AND_QUANTIZE --ds_checkpoint ./checkpoints/global_step100 --hf_out_dir ./my_hf_model --awq_out_dir ./my_hf_model_awq_4bit

echo "=== ALL LIFECYCLE STAGES COMPLETED SUCCESSFULLY ==="
