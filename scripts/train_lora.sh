#!/usr/bin/env bash
# Step 2 - LoRA training: fine-tune Qwen-Image 2.1 on the MNIST edit tasks with the vendored diffusers img2img trainer.
# Deck: p.3 (One training step), p.7 (Where the LoRA goes), p.8-9 (Training loss, Loss by sigma),
#       p.13-15 (Success rate, Before vs after training, Learnable vs not learnable).
#
# Defaults are exactly experiment B: rank 32 (alpha 32), attention + image-MLP layers, lr 1e-4 constant (no warmup),
# 3000 steps, batch 1, bf16, 512x512, sigma ~ U(0,1) with no loss weighting (--weighting_scheme none), AdamW (wd 1e-4),
# seed 0. About 1.5 h on 1x H100 80GB.
#
# Usage:
#   bash scripts/train_lora.sh MODEL_DIR DATA_DIR OUTPUT_DIR [options]
#     MODEL_DIR   local Qwen/Qwen-Image-2.1 snapshot (see README: hf download ... --revision 790c926...)
#     DATA_DIR    output of scripts/prepare_data.py (default 4 x 500 = 2000 pairs)
#     OUTPUT_DIR  receives ckpt-<step>/pytorch_lora_weights.safetensors, checkpoint-<step>/ (accelerate state),
#                 hook_logs/ (train.jsonl, probe.jsonl, val.jsonl, meta.json, val images), FINAL.json
# Options:
#   --train-shift S   train-time sigma shift: sigma' = S*sigma / (1 + (S-1)*sigma); default 1 = off (experiment E used 5)
#   --next-target T   next targets of the fixed test set (random | proto); default: read DATA_DIR/summary.json so it
#                     matches the training data (experiment F)
#   --steps N         total steps (default 3000)          --ckpt-every N   save the LoRA every N steps (default 500)
#   --val-every N     sample the test set every N steps (default 200)
#   --probe-every N   held-out probe loss every N steps (default 50)
#   --n-val N         test pairs per task for validation / probe (default 16)
#   --resume latest   resume from the newest checkpoint-<step>/ in OUTPUT_DIR
#   --smoke           smoke test: 6 steps, validation / probe every 3 steps, 2 pairs per task (a few minutes)
# The old env overrides (STEPS=... CKPT_EVERY=... VAL_EVERY=... PROBE_EVERY=... N_VAL=... RESUME=...) still work;
# command-line options take precedence.
#
# Examples:
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_train outputs/lora_smoke --smoke
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_train outputs/lora_b                  # experiment B
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_train outputs/lora_e --train-shift 5  # experiment E
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_proto outputs/lora_f                  # experiment F
set -euo pipefail

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit "${1:-1}"; }
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage 0
[[ $# -lt 3 ]] && usage 1
MODEL_DIR="$1"; DATA_DIR="$2"; OUTPUT_DIR="$3"; shift 3

STEPS="${STEPS:-3000}"; CKPT_EVERY="${CKPT_EVERY:-500}"; VAL_EVERY="${VAL_EVERY:-200}"
PROBE_EVERY="${PROBE_EVERY:-50}"; N_VAL="${N_VAL:-16}"; RESUME="${RESUME:-}"
TRAIN_SHIFT=1; NEXT_TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-shift) TRAIN_SHIFT="$2"; shift 2 ;;
    --next-target) NEXT_TARGET="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --ckpt-every) CKPT_EVERY="$2"; shift 2 ;;
    --val-every) VAL_EVERY="$2"; shift 2 ;;
    --probe-every) PROBE_EVERY="$2"; shift 2 ;;
    --n-val) N_VAL="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --smoke) STEPS=6; CKPT_EVERY=6; VAL_EVERY=3; PROBE_EVERY=3; N_VAL=2; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

# The test set's next targets must match the training data: unless given, read prepare_data.py's summary.json.
if [[ -z "$NEXT_TARGET" ]]; then
  PY="$(command -v python || command -v python3)"
  NEXT_TARGET=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('next_target','random'))" \
    "$DATA_DIR/summary.json" 2>/dev/null || echo random)
fi
[[ "$NEXT_TARGET" == "random" || "$NEXT_TARGET" == "proto" ]] || { echo "--next-target must be random|proto" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTPUT_DIR"
echo "train_lora: steps=$STEPS train_shift=$TRAIN_SHIFT next_target=$NEXT_TARGET data=$DATA_DIR -> $OUTPUT_DIR"

export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HOOK_OUT="${HOOK_OUT:-$OUTPUT_DIR/hook_logs}"
export HOOK_N_VAL="$N_VAL" HOOK_VAL_EVERY="$VAL_EVERY" HOOK_PROBE_EVERY="$PROBE_EVERY" HOOK_CKPT_EVERY="$CKPT_EVERY"
export HOOK_NEXT_TARGET="$NEXT_TARGET" HOOK_DATA_SUMMARY="$DATA_DIR/summary.json"

RESUME_ARG=()
[[ -n "$RESUME" ]] && RESUME_ARG=(--resume_from_checkpoint "$RESUME")

# --instance_prompt is required by the upstream argparse but unused: every row has its own caption.
# --random_flip is deliberately NOT set: a flip would change the meaning of the rotation tasks.
# Batch size must stay 1: the four instructions have different lengths / image-pad positions, and the
# trainer refuses to batch mismatched prompt layouts.
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  "$REPO/train/train_dreambooth_lora_qwenimage21_img2img.py" \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --dataset_name "$DATA_DIR" --cond_image_column cond_image --image_column image --caption_column caption \
  --instance_prompt "把图片旋转 180 度" \
  --output_dir "$OUTPUT_DIR" --mixed_precision bf16 --resolution 512 --center_crop \
  --train_batch_size 1 --gradient_accumulation_steps 1 \
  --rank 32 --lora_alpha 32 \
  --lora_layers "to_k,to_q,to_v,to_out.0,img_mlp.proj,img_mlp.gate_layer,img_mlp.out" \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$STEPS" --checkpointing_steps "$CKPT_EVERY" --checkpoints_total_limit 2 \
  --weighting_scheme none --train_shift "$TRAIN_SHIFT" \
  --optimizer AdamW --adam_weight_decay 1e-4 --max_grad_norm 1.0 \
  --cache_latents --allow_tf32 --seed 0 --report_to tensorboard \
  --validation_num_inference_steps 40 \
  ${RESUME_ARG[@]+"${RESUME_ARG[@]}"}
