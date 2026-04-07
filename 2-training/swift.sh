#!/usr/bin/env bash
set -euo pipefail



# python train_qwen3_rerankers.py --config config.yaml --only qwen3-reranker-4b
# python train_qwen3_rerankers.py --config config.yaml

# =========================
# User settings
# =========================
TRAIN_FILE="${TRAIN_FILE:-data/swift_train.jsonl}"
VALID_FILE="${VALID_FILE:-data/swift_valid.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

# Training style:
# listwise is usually the best fit for grouped candidate ranking.
LOSS_TYPE="${LOSS_TYPE:-listwise_reranker}"

# LoRA settings
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Common training settings
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEED="${SEED:-42}"

# Per-model batch sizes
BS_06B="${BS_06B:-2}"
BS_4B="${BS_4B:-1}"
BS_8B="${BS_8B:-1}"

# If you have multiple GPUs, set CUDA_VISIBLE_DEVICES before running.
# Example:
# CUDA_VISIBLE_DEVICES=0,1 bash train_qwen3_rerankers.sh

mkdir -p "${OUTPUT_ROOT}"

# SWIFT expands each item into grouped positives/negatives.
# These env vars are documented by SWIFT for reranker training.
export MAX_POSITIVE_SAMPLES="${MAX_POSITIVE_SAMPLES:-1}"
export MAX_NEGATIVE_SAMPLES="${MAX_NEGATIVE_SAMPLES:-7}"
export LISTWISE_RERANKER_TEMPERATURE="${LISTWISE_RERANKER_TEMPERATURE:-1.0}"
export LISTWISE_RERANKER_MIN_GROUP_SIZE="${LISTWISE_RERANKER_MIN_GROUP_SIZE:-2}"
export GENERATIVE_RERANKER_POSITIVE_TOKEN="${GENERATIVE_RERANKER_POSITIVE_TOKEN:-yes}"
export GENERATIVE_RERANKER_NEGATIVE_TOKEN="${GENERATIVE_RERANKER_NEGATIVE_TOKEN:-no}"

train_one () {
  local MODEL_NAME="$1"
  local RUN_NAME="$2"
  local BATCH_SIZE="$3"

  local OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${OUT_DIR}"

  echo "========================================================"
  echo "Training ${MODEL_NAME}"
  echo "Output: ${OUT_DIR}"
  echo "========================================================"

  swift sft \
    --model "${MODEL_NAME}" \
    --task_type generative_reranker \
    --loss_type "${LOSS_TYPE}" \
    --train_type lora \
    --tuner_backend peft \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
    --dataset "${TRAIN_FILE}" \
    --val_dataset "${VALID_FILE}" \
    --dataset_num_proc 4 \
    --max_length "${MAX_LENGTH}" \
    --learning_rate "${LR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --eval_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps "${LOGGING_STEPS}" \
    --seed "${SEED}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --gradient_checkpointing true \
    --output_dir "${OUT_DIR}"

  echo "Finished ${MODEL_NAME}"
  echo "Checkpoints saved under ${OUT_DIR}"
}

train_one "Qwen/Qwen3-Reranker-0.6B" "qwen3-reranker-0.6b" "${BS_06B}"
train_one "Qwen/Qwen3-Reranker-4B"   "qwen3-reranker-4b"   "${BS_4B}"
train_one "Qwen/Qwen3-Reranker-8B"   "qwen3-reranker-8b"   "${BS_8B}"