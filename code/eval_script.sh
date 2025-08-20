#!/bin/bash
# ============================================================
# LoRA Fine-Tuning / Evaluation / Prediction Runner
# ============================================================
export CUDA_VISIBLE_DEVICES=0

FOLDER_NAME="exp_mtl_07"
NUM_EXAMPLES=-1
ENABLE_LORA="True"

MODEL_NAME="google/gemma-2-2b-it"  # "google/gemma-3-1b-it" or "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/${FOLDER_NAME}/lora_model"
PREDICT_CSV="/home/kaushal.maurya/tutor_eval_auto_metrics/data_preprocessing/output/MTL_test.csv"
OUTPUT_DIR="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/${FOLDER_NAME}"

python eval_tutor_response_assessment.py \
    --model_name "$MODEL_NAME" \
    --adapter_path "$ADAPTER_PATH" \
    --eval_csv "$PREDICT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --num_examples "$NUM_EXAMPLES" \
    --enable_lora "$ENABLE_LORA" \

