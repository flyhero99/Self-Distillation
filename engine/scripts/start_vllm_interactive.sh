#!/bin/bash
# Interactive vLLM Server Script for OSC
#
# Usage:
#   # First get an interactive GPU session with 2 GPUs: 
#   srun --gpus=2 --mem=64G --time=8:00:00 --account=PAA0201 --pty bash
#   srun --gpus=4 --mem=200G --time=96:00:00 --account=PAA0201 --pty bash
#   # Then run this script:
#   bash engine/scripts/start_vllm_interactive.sh
#
# Options (all optional):
#   bash start_vllm_interactive.sh [MODEL] [PORT] [NUM_GPUS] [MAX_LEN]
#
# Examples:
#   bash engine/scripts/start_vllm_interactive.sh                           # Defaults: Qwen3-32B, port 8000, auto GPUs
#   bash engine/scripts/start_vllm_interactive.sh Qwen/Qwen3-4B 8000 2
#   bash engine/scripts/start_vllm_interactive.sh mistralai/Mistral-7B-Instruct-v0.3 8000 1

set -e

# ============================================
# Cleanup any stale vLLM processes
# ============================================
pkill -f "vllm.entrypoints" 2>/dev/null && echo "Cleaned up stale vLLM processes" || echo "No vLLM process to cleanup"

# ============================================
# Load OSC vLLM module
# ============================================
module load vllm/0.13.0

# ============================================
# Set HuggingFace cache to user directory
# (Avoids permission issues with shared cache)
# ============================================
export HF_HOME=/fs/ess/PAA0201/hananemoussa/.cache/huggingface
mkdir -p $HF_HOME

# ============================================
# NCCL settings for multi-GPU stability
# ============================================
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# ============================================
# Detect available GPUs
# ============================================
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ============================================
# Configuration (with defaults)
# ============================================
MODEL_NAME=${1:-"Qwen/Qwen3-32B"}
PORT=${2:-8000}
TENSOR_PARALLEL=${3:-$NUM_GPUS}  # Default to all available GPUs
MAX_MODEL_LEN=${4:-16384}  # Reduced from 32768 to lower memory pressure
GPU_MEMORY_UTILIZATION=0.90  # Reduced from 0.95 to leave more headroom

# Safety check: tensor parallel can't exceed available GPUs
if [ "$TENSOR_PARALLEL" -gt "$NUM_GPUS" ]; then
    echo "WARNING: Requested $TENSOR_PARALLEL GPUs but only $NUM_GPUS available. Using $NUM_GPUS."
    TENSOR_PARALLEL=$NUM_GPUS
fi

# Check if we have enough GPUs for Qwen3-32B
if [[ "$MODEL_NAME" == *"Qwen3-32B"* ]] && [ "$NUM_GPUS" -lt 2 ]; then
    echo ""
    echo "ERROR: Qwen3-32B requires 2x A100 40GB GPUs but only $NUM_GPUS GPU(s) detected."
    echo ""
    echo "Please request more GPUs:"
    echo "  srun --gpus=2 --mem=64G --time=4:00:00 --account=PAA0201 --pty bash"
    echo ""
    exit 1
fi

echo "=============================================="
echo "vLLM Interactive Server (OSC)"
echo "=============================================="
echo "Hostname: $(hostname)"
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "Available GPUs: $NUM_GPUS"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "=============================================="
echo ""
echo "Server will be available at:"
echo "  http://$(hostname):$PORT/v1"
echo ""
echo "To run inference, open a NEW terminal and run:"
echo ""
echo "  cd /fs/ess/PAA0201/hananemoussa/my-SAB"
echo "  export VLLM_API_BASE=http://$(hostname):$PORT/v1"
echo "  conda activate sci-agent"
echo "  python -u run_infer.py --llm_engine_name \"vllm:$MODEL_NAME\" \\"
echo "      --log_fname logs/qwen3_4b_thinking_enabled.jsonl \\"
echo "      --out_fpath pred_programs_qwen3_4b_thinking_enabled/"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="
echo ""

# ============================================
# Start vLLM server with stability flags
# ============================================
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --dtype bfloat16 \
    --enforce-eager \
    --disable-custom-all-reduce \
    --disable-log-requests \
    --reasoning-parser deepseek_r1
