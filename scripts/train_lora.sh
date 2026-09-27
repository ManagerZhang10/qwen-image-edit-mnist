#!/usr/bin/env bash
# LoRA fine-tuning of Qwen-Image 2.1 on the MNIST edit tasks, with the vendored diffusers img2img trainer
# (train/train_dreambooth_lora_qwenimage21_img2img.py @ diffusers e0abab8, plus 4 logging hooks).
#
# Hyperparameters are exactly those of experiment B:
#   rank 32 (alpha 32), attention + image-MLP layers, lr 1e-4 constant (no warmup), 3000 steps, batch 1,
#   bf16, 512x512, sigma ~ U(0,1) with no loss weighting (--weighting_scheme none), AdamW (wd 1e-4), seed 0.
#
# Usage:
#   bash scripts/train_lora.sh MODEL_DIR DATA_DIR OUTPUT_DIR
#     MODEL_DIR   local Qwen/Qwen-Image-2.1 snapshot (see README: hf download ... --revision 790c926...)
#     DATA_DIR    output of scripts/prepare_data.py (default n-per-task 500 -> 2000 pairs)
#     OUTPUT_DIR  receives ckpt-<step>/pytorch_lora_weights.safetensors, checkpoint-<step>/ (accelerate
#                 state), hook_logs/ (train.jsonl, probe.jsonl, val.jsonl, meta.json, val images), FINAL.json
#
# Overrides (env): STEPS=3000  CKPT_EVERY=500  VAL_EVERY=200  PROBE_EVERY=50  N_VAL=16
#                  RESUME=latest   (pass --resume_from_checkpoint latest)
# Smoke test:      STEPS=6 CKPT_EVERY=6 VAL_EVERY=3 PROBE_EVERY=3 N_VAL=2 bash scripts/train_lora.sh ...
set -euo pipefail

if [[ $# -ne 3 ]]; then
  sed -n '2,20p' "$0"; exit 1
fi
MODEL_DIR="$1"; DATA_DIR="$2"; OUTPUT_DIR="$3"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export HOOK_OUT="${HOOK_OUT:-$OUTPUT_DIR/hook_logs}"
export HOOK_N_VAL="${N_VAL:-16}" HOOK_VAL_EVERY="${VAL_EVERY:-200}" HOOK_PROBE_EVERY="${PROBE_EVERY:-50}"
export HOOK_CKPT_EVERY="${CKPT_EVERY:-500}"

RESUME_ARG=()
[[ -n "${RESUME:-}" ]] && RESUME_ARG=(--resume_from_checkpoint "$RESUME")

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
  --max_train_steps "${STEPS:-3000}" --checkpointing_steps "${CKPT_EVERY:-500}" --checkpoints_total_limit 2 \
  --weighting_scheme none --optimizer AdamW --adam_weight_decay 1e-4 --max_grad_norm 1.0 \
  --cache_latents --allow_tf32 --seed 0 --report_to tensorboard \
  --validation_num_inference_steps 40 \
  ${RESUME_ARG[@]+"${RESUME_ARG[@]}"}
