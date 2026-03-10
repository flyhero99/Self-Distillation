#!/bin/bash
###############################################################################
# Interactive vLLM Server for Qwen3-4B-Instruct Full Fine-tuned
#
# Usage Option 1 - All-in-one (requests GPUs and starts server):
#   bash start_vllm_qwen3_4b_instruct_full_interactive.sh
#
# Usage Option 2 - Manual (more control):
#   Step 1: Request interactive session with GPUs
#     srun --account=PAS3019 --partition=quad --gres=gpu:a100:2 \
#          --cpus-per-task=16 --mem=64G --time=6:00:00 --pty bash
#
#   Step 2: In the interactive session, run:
#     cd /fs/ess/PAA0201/flyhero/my-SAB
#     bash start_vllm_qwen3_4b_instruct_full_interactive.sh run_server_only
#
# The server will run in the foreground. Press Ctrl+C to stop it.
###############################################################################

set -e

# ============================================
# Mode selection
# ============================================
if [ "$1" == "run_server_only" ]; then
    # Skip srun, just run the server (assumes we're already in an interactive session)
    RUN_SERVER_ONLY=true
else
    # Launch interactive session first
    RUN_SERVER_ONLY=false
fi

# ============================================
# Configuration
# ============================================
MODEL_PATH="/fs/scratch/PAS3019/yifeili/Self-Distillation/output_qwen3-4b-instruct-2507/checkpoint-501"
PORT=${PORT:-8001}
NUM_GPUS=2
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.90

# ============================================
# Request interactive session (if needed)
# ============================================
if [ "$RUN_SERVER_ONLY" = false ]; then
    echo "=============================================="
    echo "Requesting Interactive GPU Session"
    echo "=============================================="
    echo "Account: PAS3019"
    echo "Partition: quad"
    echo "GPUs: 2x A100"
    echo "Time: 6 hours"
    echo "=============================================="
    echo ""
    echo "Once you get the session, the server will start automatically."
    echo "Press Ctrl+C to stop the server when done."
    echo ""

    # Launch interactive session and run this script in server-only mode
    srun --account=PAS3019 \
         --partition=quad \
         --gres=gpu:a100:2 \
         --cpus-per-task=16 \
         --mem=64G \
         --time=2:00:00 \
         --pty bash -c "cd /fs/ess/PAA0201/flyhero/my-SAB && bash start_vllm_qwen3_4b_instruct_full_interactive.sh run_server_only"

    exit 0
fi

# ============================================
# Server-only mode (runs in interactive session)
# ============================================

echo "=============================================="
echo "Setting up vLLM Server"
echo "=============================================="

# Cleanup any stale vLLM processes
pkill -f "vllm.entrypoints" 2>/dev/null && echo "✓ Cleaned up stale vLLM processes" || echo "✓ No stale processes"

# Setup environment
module load vllm/0.13.0

# Set HuggingFace cache
export HF_HOME=/users/PAA0201/flyhero/.cache/huggingface
mkdir -p $HF_HOME

# NCCL settings for multi-GPU stability
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Detect and configure GPUs
DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "✓ Detected $DETECTED_GPUS GPU(s)"

if [ "$DETECTED_GPUS" -lt "$NUM_GPUS" ]; then
    echo "WARNING: Expected $NUM_GPUS GPUs but only found $DETECTED_GPUS"
    echo "Using $DETECTED_GPUS GPUs instead"
    NUM_GPUS=$DETECTED_GPUS
fi

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "✓ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

HOSTNAME=$(hostname)

echo ""
echo "=============================================="
echo "vLLM Server - Qwen3-4B-Instruct Full (Interactive)"
echo "=============================================="
echo "Hostname: $HOSTNAME"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel Size: $NUM_GPUS"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "=============================================="
echo ""
echo "Server will be available at:"
echo "  http://${HOSTNAME}:${PORT}/v1"
echo ""
echo "To run inference, open a NEW terminal and run:"
echo ""
echo "  cd /fs/ess/PAA0201/flyhero/my-SAB"
echo "  bash run_inference_qwen3_4b_instruct_full.sh $HOSTNAME $PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

# Start vLLM server (runs in foreground)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$NUM_GPUS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --dtype bfloat16 \
    --enforce-eager \
    --disable-custom-all-reduce \
    --disable-log-requests
