#!/bin/bash
# ============================================================
# LoRA Fine-Tuning Runner
# ============================================================

export CUDA_VISIBLE_DEVICES=1

MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/inputs"
OUTPUT_DIR="/home/kaushal.maurya/tutor_eval_auto_metrics/lora_mtl/outputs/exp_09"
MAX_LENGTH=2048
BATCH_SIZE=4
GRAD_ACCUM=1
EPOCHS=10
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01
LOGGING_STEPS=100
SAVE_STEPS=500
EVAL_STEPS=500

# LoRA hyperparameters
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Data Augmentation
OVERSAMPLE_METHOD="random" # Options: random, smote, adasyn

# List of dimensions to train on (space-separated)
DIMENSIONS=("Mistake_Identification" "Mistake_Location" "Providing_Guidance" "Actionability")

mkdir -p "$OUTPUT_DIR"

# Run training
python -u train.py \
    --train \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --dimensions "${DIMENSIONS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --oversample_method $OVERSAMPLE_METHOD \
    # --include_label_definitions \