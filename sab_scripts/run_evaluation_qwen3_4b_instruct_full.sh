#!/bin/bash
###############################################################################
# Evaluation Script for Qwen3-4B-Instruct Full Fine-tuned Model
#
# Prerequisites:
#   1. Inference must be completed (pred_programs_qwen3_4b_instruct_full/ exists)
#   2. sci-agent conda environment must be available
#
# Usage:
#   bash run_evaluation_qwen3_4b_instruct_full.sh
###############################################################################

set -e

echo "=============================================="
echo "Qwen3-4B-Instruct Full Model Evaluation"
echo "=============================================="
echo "Input: pred_programs_qwen3_4b_instruct_full/"
echo "Output: pred_results_qwen3_4b_instruct_full/"
echo "Log: logs/qwen3_4b_instruct_full_eval.jsonl"
echo "=============================================="
echo ""

# Check prerequisites
if [ ! -d "pred_programs_qwen3_4b_instruct_full" ]; then
    echo "Error: pred_programs_qwen3_4b_instruct_full/ directory not found"
    echo "Please run inference first: bash run_inference_qwen3_4b_instruct_full.sh <HOSTNAME>"
    exit 1
fi

# Count predicted programs
NUM_PROGRAMS=$(ls pred_programs_qwen3_4b_instruct_full/pred_*.py 2>/dev/null | wc -l)
if [ $NUM_PROGRAMS -eq 0 ]; then
    echo "Error: No predicted programs found in pred_programs_qwen3_4b_instruct_full/"
    echo "Please run inference first"
    exit 1
fi

echo "Found $NUM_PROGRAMS predicted programs to evaluate"
echo ""

# Setup environment
module load miniconda3/24.1.2-py310
source activate /users/PAA0201/hananemoussa/.conda/envs/sci-agent 2>/dev/null || conda activate sci-agent

# Create output directories
mkdir -p logs
mkdir -p pred_results_qwen3_4b_instruct_full

echo "Starting evaluation..."
echo "This may take a while depending on the number of programs..."
echo ""

# Run evaluation
python -u run_eval.py \
    --pred_program_path pred_programs_qwen3_4b_instruct_full/ \
    --result_path pred_results_qwen3_4b_instruct_full/ \
    --log_fname logs/qwen3_4b_instruct_full_eval.jsonl \
    --gold_program_path benchmark/gold_programs/ \
    --eval_program_path benchmark/eval_programs/

echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "=============================================="
echo "Results saved to: pred_results_qwen3_4b_instruct_full/"
echo "Log saved to: logs/qwen3_4b_instruct_full_eval.jsonl"
echo ""
echo "To view results summary:"
echo "  python compute_scores.py logs/qwen3_4b_instruct_full_eval.jsonl"
