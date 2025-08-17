#!/bin/bash
# ============================================================
# LoRA Fine-Tuning / Evaluation / Prediction Runner
# ============================================================
MODEL_NAME="google/gemma-2-2b-it" #"google/gemma-3-1b-it" #"meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/outputs_llama_exp02/lora_model"
PREDICT_CSV="/home/kaushal.maurya/tutor_eval_auto_metrics/data_preprocessing/output/MI_test.csv"
# OUTPUT_DIR="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/outputs_llama_exp01"

python eval_tutor_response_assessment.py \
    --model_name $MODEL_NAME \
    --adapter_path $ADAPTER_PATH \
    --eval_csv $PREDICT_CSV \
    # --output_dir $OUTPUT_DIR
