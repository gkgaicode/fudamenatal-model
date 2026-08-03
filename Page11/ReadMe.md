Here is the final system component: a production-grade, multi-node cluster orchestration script (slurm_cluster_submit.sh) paired with an optimized local fallback executor (local_cluster_run.sh).
These scripts manage your machine networking, configure environment variables for distributed communication, and launch your agent pipeline across thousands of compute cores.
------------------------------
## 1. Enterprise Option: SLURM Cluster Submission Script (slurm_cluster_submit.sh)
If you are deploying on a professional supercomputing cluster (such as AWS ParallelCluster, Azure CycleCloud, or on-premise hardware) using the SLURM workload manager, use this script. It requests resources, calculates network nodes, and provisions the torchrun execution path automatically.

#!/bin/bash#SBATCH --job-name=llm_foundational_agent  # Job submission tracking designation#SBATCH --nodes=4                          # Request total number of physical nodes/machines#SBATCH --ntasks-per-node=1                # One launch task per node (handled internally by torchrun)#SBATCH --gres=gpu:8                       # Request 8 GPUs per machine (e.g., 4 Nodes x 8 GPUs = 32 GPUs total)#SBATCH --cpus-per-task=64                 # CPU cores allocated per node for data loading operations#SBATCH --mem=0                            # Request unlimited system RAM per node#SBATCH --time=48:00:00                    # Wallclock maximum limit (Hours:Minutes:Seconds)#SBATCH --output=logs/cluster_run_%j.out   # Standard stdout execution logs logging path#SBATCH --error=logs/cluster_run_%j.err    # Standard stderr crash dumps logging path
# --- 1. Infrastructure Architecture Network Capture ---# Grab the primary address of Node 0 to establish the master orchestration rendezvous pointer
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500 # Open a protected high-range port for cross-gpu sync operations
# Calculate cluster distribution properties dynamically from SLURM environment allocations
export WORLD_SIZE=$((SLURM_NNODES * 8)) # Total number of active physical GPUs in play
export GPUS_PER_NODE=8

echo "=== INITIALIZING MULTI-NODE DISTRIBUTED AGENT HARDWARE LAYER ==="
echo "Master Rendezvous Address: $MASTER_ADDR"
echo "Allocated Cluster Nodes: $SLURM_NNODES"
echo "Total Global GPU World Size: $WORLD_SIZE"
echo "================================================================"
# Create target logs tracing directory if missing
mkdir -p logs
# --- 2. Multi-Node Execution Dispatch Engine ---# srun clones this exact launch array across all requested physical server racks concurrently
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    agent.py --stage PRETRAIN_OR_SFT --num_gpus=$WORLD_SIZE --ds_config ds_config_zero3.json

------------------------------
## 2. Local Option: Multi-GPU Standalone Executor (local_cluster_run.sh)
If you are training on a single, high-spec server containing multiple GPUs (such as an NVIDIA DGX H100 workstation) rather than a multi-node server cluster, use this streamlined local executor.

#!/bin/bash
# Terminate execution paths cleanly if any individual sub-command throws a failure flagset -e
# --- 1. Local Device Auditing ---# Query system configurations to find out exactly how many active GPUs are inside this machine
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

------------------------------
## How to Run Your System
Depending on your hardware setup, apply file-system execution privileges to the appropriate script and launch your pipeline:
## To Run on a SLURM Cluster:

chmod +x slurm_cluster_submit.sh
sbatch slurm_cluster_submit.sh

## To Run Locally on a Single Machine:

chmod +x local_cluster_run.sh
./local_cluster_run.sh

Your autonomous LLM foundational architecture codebase is now fully complete, locked to industry-grade versions, covered by validation testing suites, and ready for high-performance deployment.
If you plan to scale this configuration up or migrate your system live, let me know if you would like me to generate a Grafana / Prometheus telemetry tracking profile to visually monitor your GPU temperatures, VRAM consumption, and training loss charts in real time!
