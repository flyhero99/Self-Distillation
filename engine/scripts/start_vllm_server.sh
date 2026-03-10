#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --output=vllm_server_%j.log
#SBATCH --error=vllm_server_%j.log
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=PAA0201

# vLLM Server SLURM Job Script for OSC
#
# Usage:
#   cd /fs/ess/PAA0201/hananemoussa/my-SAB
#   sbatch engine/scripts/start_vllm_server.sh
#
# To use a different model:
#   MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 TENSOR_PARALLEL=1 sbatch engine/scripts/start_vllm_server.sh

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
# Configuration (with defaults)
# ============================================
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}
PORT=${PORT:-8000}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}  # Reduced from 32768 to lower memory pressure
GPU_MEMORY_UTILIZATION=0.90  # Reduced from 0.95 to leave more headroom

echo "=============================================="
echo "vLLM Server (SLURM Job)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "=============================================="
echo ""
echo "Server will be available at:"
echo "  http://$(hostname):$PORT/v1"
echo ""
echo "To run inference from login node:"
echo ""
echo "  cd /fs/ess/PAA0201/hananemoussa/my-SAB"
echo "  export VLLM_API_BASE=http://$(hostname):$PORT/v1"
echo "  python -u run_infer.py --llm_engine_name \"vllm:$MODEL_NAME\" \\"
echo "      --log_fname logs/qwen3_32b.jsonl \\"
echo "      --out_fpath pred_programs_qwen3/"
echo ""
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
    --disable-log-requests
