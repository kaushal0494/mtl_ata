#!/bin/bash
# ============================================================
# LoRA Fine-Tuning / Evaluation / Prediction Runner
# ============================================================
MODEL_NAME="google/gemma-2-2b-it" #"meta-llama/Llama-3.1-8B-Instruct" #"google/gemma-3-1b-it"
MODEL_PATH="./outputs/MI_lora_model"
TRAIN_CSV="/home/kaushal.maurya/tutor_eval_auto_metrics/data_preprocessing/output/MI_train.csv"
EVAL_CSV="/home/kaushal.maurya/tutor_eval_auto_metrics/data_preprocessing/output/MI_dev.csv"
PREDICT_CSV="/home/kaushal.maurya/tutor_eval_auto_metrics/data_preprocessing/output/MI_test.csv"
OUTPUT_DIR="./outputs_llama_exp03"
MAX_LENGTH=1024
BATCH_SIZE=4
GRAD_ACCUM=1
EPOCHS=3
LEARNING_RATE=5e-4
WARMUP_STEPS=100
WEIGHT_DECAY=0.01

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="run_${TIMESTAMP}.log"

MODE=$1  # train / eval / predict

if [ "$MODE" = "train" ]; then
    python new_tutor_response_assessment.py \
        --train \
        --model_name "$MODEL_NAME" \
        --train_csv "$TRAIN_CSV" \
        --eval_csv "$EVAL_CSV" \
        --output_dir "$OUTPUT_DIR" \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --warmup_steps $WARMUP_STEPS \
        --weight_decay $WEIGHT_DECAY \
        2>&1 | tee "$OUTPUT_DIR/$LOG_FILE"

elif [ "$MODE" = "eval" ]; then
    python tutor_response_assessment.py \
        --eval_only \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --eval_csv "$PREDICT_CSV" \
        --max_length $MAX_LENGTH \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/$LOG_FILE"

elif [ "$MODE" = "predict" ]; then
    python tutor_response_assessment.py \
        --predict_csv "$PREDICT_CSV" \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --max_length $MAX_LENGTH \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/$LOG_FILE"

else
    echo "Usage: bash run_lora.sh [train|eval|predict]"
    exit 1
fi
