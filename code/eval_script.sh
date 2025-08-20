#!/bin/bash
# ============================================================
# LoRA Evaluation & Prediction Runner
# ============================================================
export CUDA_VISIBLE_DEVICES=1

FOLDER_NAME="exp_06"
NUM_EXAMPLES=-1
ENABLE_LORA="True"
MAX_LENGTH=2048

MODEL_NAME="google/gemma-2-2b-it"  # "google/gemma-3-1b-it" or "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/outputs/${FOLDER_NAME}/lora_model"
PREDICT_CSV="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/inputs/MTL_test.csv"
OUTPUT_DIR="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/outputs/${FOLDER_NAME}"

# Generation parameters
TEMPERATURE=1.0
TOP_K=50
TOP_P=1.0

python evaluation.py \
    --model_name "$MODEL_NAME" \
    --adapter_path "$ADAPTER_PATH" \
    --eval_csv "$PREDICT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --num_examples "$NUM_EXAMPLES" \
    --enable_lora "$ENABLE_LORA" \
    --max_length "$MAX_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --include_label_definitions \
    # --do_sample \
    