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
MODEL_PATH="/fs/scratch/PAS3019/yifeili/Self-Distillation/output_qwen3-4b-instruct-2507/checkpoint-501"
RUN_TAG="qwen3_4b_instruct_ckpt501"

echo "=============================================="
echo "Qwen3-4B-Instruct Checkpoint-501 Inference"
echo "=============================================="
echo "vLLM Server: http://${HOSTNAME}:${PORT}/v1"
echo "Model: ${MODEL_PATH}"
echo "Output: pred_programs_${RUN_TAG}/"
echo "Log: logs/${RUN_TAG}_infer.jsonl"
echo "=============================================="
echo ""

# Setup environment
# Prefer user's own conda installation to avoid module/env mismatch.
if [ -f /users/PAA0201/flyhero/miniconda3/etc/profile.d/conda.sh ]; then
    source /users/PAA0201/flyhero/miniconda3/etc/profile.d/conda.sh
else
    module load miniconda3/24.1.2-py310
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate sci-agent

# Create output directories
mkdir -p logs
mkdir -p "pred_programs_${RUN_TAG}"

# Set vLLM API endpoint
export VLLM_API_BASE="http://${HOSTNAME}:${PORT}/v1"
export VLLM_ENABLE_THINKING="true"  # Disable Qwen3 thinking mode (trained without thinking)

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
    --llm_engine_name "vllm:${MODEL_PATH}" \
    --log_fname "logs/${RUN_TAG}_infer.jsonl" \
    --out_fpath "pred_programs_${RUN_TAG}/" \
    --context_cutoff 16000

echo ""
echo "=============================================="
echo "Inference completed!"
echo "=============================================="
echo "Programs saved to: pred_programs_${RUN_TAG}/"
echo "Log saved to: logs/${RUN_TAG}_infer.jsonl"
echo ""
echo "To run evaluation, use:"
echo "  bash run_evaluation_qwen3_4b_instruct_full.sh"
