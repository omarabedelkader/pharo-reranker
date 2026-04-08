#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_qwen25_coder_fim_lora.sh
#
# Override any variable inline, for example:
#   MODEL_NAME=Qwen/Qwen2.5-Coder-3B MAX_LENGTH=2048 bash run_qwen25_coder_fim_lora.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-1.5B}"
DATA_DIR="${DATA_DIR:-./data}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.jsonl}"
VALIDATION_FILE="${VALIDATION_FILE:-${DATA_DIR}/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
TRAIN_BATCH="${TRAIN_BATCH:-2}"
EVAL_BATCH="${EVAL_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SEED="${SEED:-42}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SPLIT_RATIO="${SPLIT_RATIO:-0.02}"
USE_4BIT="${USE_4BIT:-1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
TRAIN_ON_PROMPT="${TRAIN_ON_PROMPT:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-/mnt/data/qwen25_coder_fim_lora.py}"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_PATH" train
  --model-name "$MODEL_NAME"
  --data-dir "$DATA_DIR"
  --train-file "$TRAIN_FILE"
  --output-dir "$OUTPUT_DIR"
  --max-length "$MAX_LENGTH"
  --per-device-train-batch-size "$TRAIN_BATCH"
  --per-device-eval-batch-size "$EVAL_BATCH"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --learning-rate "$LR"
  --num-train-epochs "$EPOCHS"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-ratio "$WARMUP_RATIO"
  --logging-steps "$LOGGING_STEPS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --seed "$SEED"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --split-ratio "$SPLIT_RATIO"
)

if [[ -n "$VALIDATION_FILE" ]]; then
  cmd+=(--validation-file "$VALIDATION_FILE")
fi

if [[ "$USE_4BIT" == "1" ]]; then
  cmd+=(--use-4bit)
else
  cmd+=(--no-use-4bit)
fi

if [[ "$BF16" == "1" ]]; then
  cmd+=(--bf16)
fi

if [[ "$FP16" == "1" ]]; then
  cmd+=(--fp16)
fi

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  cmd+=(--gradient-checkpointing)
fi

if [[ "$TRAIN_ON_PROMPT" == "1" ]]; then
  cmd+=(--train-on-prompt)
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
"${cmd[@]}"
