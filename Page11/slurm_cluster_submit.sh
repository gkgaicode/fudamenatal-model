#!/bin/bash
#SBATCH --job-name=llm_foundational_agent  # Job submission tracking designation
#SBATCH --nodes=4                          # Request total number of physical nodes/machines
#SBATCH --ntasks-per-node=1                # One launch task per node (handled internally by torchrun)
#SBATCH --gres=gpu:8                       # Request 8 GPUs per machine (e.g., 4 Nodes x 8 GPUs = 32 GPUs total)
#SBATCH --cpus-per-task=64                 # CPU cores allocated per node for data loading operations
#SBATCH --mem=0                            # Request unlimited system RAM per node
#SBATCH --time=48:00:00                    # Wallclock maximum limit (Hours:Minutes:Seconds)
#SBATCH --output=logs/cluster_run_%j.out   # Standard stdout execution logs logging path
#SBATCH --error=logs/cluster_run_%j.err    # Standard stderr crash dumps logging path

# --- 1. Infrastructure Architecture Network Capture ---
# Grab the primary address of Node 0 to establish the master orchestration rendezvous pointer
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

# --- 2. Multi-Node Execution Dispatch Engine ---
# srun clones this exact launch array across all requested physical server racks concurrently
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    agent.py --stage PRETRAIN_OR_SFT --num_gpus=$WORLD_SIZE --ds_config ds_config_zero3.json
