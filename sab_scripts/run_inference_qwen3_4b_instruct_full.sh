#!/bin/bash
###############################################################################
# Inference Script for Qwen3-4B-Instruct Full Fine-tuned Model
#
# Prerequisites:
#   1. vLLM server must be running (check with: squeue -u $USER)
#   2. Note the hostname from the vLLM server log
#
# Usage:
#   bash run_inference_qwen3_4b_instruct_full.sh <HOSTNAME> [PORT]
#
# Example:
#   bash run_inference_qwen3_4b_instruct_full.sh a0008 8001
###############################################################################

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Error: Missing hostname argument"
    echo "Usage: $0 <HOSTNAME> [PORT]"
    echo ""
    echo "To find hostname, check your vLLM server session"
    exit 1
fi

HOSTNAME=$1
PORT=${2:-8001}

echo "=============================================="
echo "Qwen3-4B-Instruct Full Model Inference"
echo "=============================================="
echo "vLLM Server: http://${HOSTNAME}:${PORT}/v1"
echo "Output: pred_programs_qwen3_4b_instruct_full/"
echo "Log: logs/qwen3_4b_instruct_full_infer.jsonl"
echo "=============================================="
echo ""

# Setup environment
module load miniconda3/24.1.2-py310
source activate /users/PAA0201/hananemoussa/.conda/envs/sci-agent 2>/dev/null || conda activate sci-agent

# Create output directories
mkdir -p logs
mkdir -p pred_programs_qwen3_4b_instruct_full

# Set vLLM API endpoint
export VLLM_API_BASE="http://${HOSTNAME}:${PORT}/v1"
export VLLM_ENABLE_THINKING="false"  # Disable Qwen3 thinking mode (trained without thinking)

echo "Testing vLLM server connection..."
curl -s "${VLLM_API_BASE}/models" > /dev/null && echo "✓ Server is responding" || {
    echo "✗ Error: Cannot connect to vLLM server at ${VLLM_API_BASE}"
    echo "Please check that the vLLM server is running:"
    echo "  squeue -u $USER"
    exit 1
}

echo ""
echo "Starting inference..."
echo ""

# Run inference
# Note: vLLM uses the full model path as the model name
python -u run_infer.py \
    --llm_engine_name "vllm:/fs/ess/PAA0201/hananemoussa/autosdt-v2/saves/qwen3-4B-instruct-full-sft" \
    --log_fname logs/qwen3_4b_instruct_full_infer.jsonl \
    --out_fpath pred_programs_qwen3_4b_instruct_full/ \
    --context_cutoff 16000

echo ""
echo "=============================================="
echo "Inference completed!"
echo "=============================================="
echo "Programs saved to: pred_programs_qwen3_4b_instruct_full/"
echo "Log saved to: logs/qwen3_4b_instruct_full_infer.jsonl"
echo ""
echo "To run evaluation, use:"
echo "  bash run_evaluation_qwen3_4b_instruct_full.sh"
