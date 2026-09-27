#!/usr/bin/env bash
# 第 2 步 · 训练 LoRA：用 vendored 的 diffusers img2img 训练脚本在 MNIST 编辑任务上微调 Qwen-Image 2.1。
# 讲义：第 3 页「训练一步怎么走」、第 7 页「LoRA 挂在哪」、第 8–9 页（训练 loss）、第 13–15 页（成功率、训练前后）。
#
# 默认超参就是实验 B：rank 32（alpha 32），注意力 + 图像 MLP 层，lr 1e-4 常数（无 warmup），3000 步，batch 1，
# bf16，512×512，σ ~ U(0,1) 不加权（--weighting_scheme none），AdamW（wd 1e-4），seed 0。约 1.5 h / 1× H100 80GB。
#
# 用法：
#   bash scripts/train_lora.sh MODEL_DIR DATA_DIR OUTPUT_DIR [选项]
#     MODEL_DIR   本地 Qwen/Qwen-Image-2.1 快照（见 README：hf download ... --revision 790c926...）
#     DATA_DIR    scripts/prepare_data.py 的输出（默认 4 × 500 = 2000 对）
#     OUTPUT_DIR  写入 ckpt-<step>/pytorch_lora_weights.safetensors、checkpoint-<step>/（accelerate 状态）、
#                 hook_logs/（train.jsonl、probe.jsonl、val.jsonl、meta.json、验证图）、FINAL.json
# 选项：
#   --train-shift S   训练时 σ shift：σ' = Sσ / (1 + (S−1)σ)，默认 1 = 不 shift（实验 E 用 5）
#   --next-target T   固定测试集里 next 的目标（random | proto）；默认读 DATA_DIR/summary.json，与训练数据一致（实验 F）
#   --steps N         总步数（默认 3000）         --ckpt-every N  每 N 步存 LoRA（默认 500）
#   --val-every N     每 N 步生图验证（默认 200）  --probe-every N 每 N 步量固定测试集 loss（默认 50）
#   --n-val N         验证 / probe 每任务几张（默认 16）
#   --resume latest   从 OUTPUT_DIR 里最新的 checkpoint-<step>/ 续训
#   --smoke           冒烟测试：6 步，每 3 步验证 / probe，每任务 2 张（几分钟，用来检查环境）
# 旧的环境变量写法（STEPS=… CKPT_EVERY=… VAL_EVERY=… PROBE_EVERY=… N_VAL=… RESUME=…）仍然有效，命令行选项优先。
#
# 例：
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_train outputs/lora_smoke --smoke
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_train outputs/lora_b              # 实验 B
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_train outputs/lora_e --train-shift 5  # 实验 E
#   bash scripts/train_lora.sh models/Qwen-Image-2.1 data/mnist_edit_proto outputs/lora_f              # 实验 F
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

# next 的测试集目标必须和训练数据一致：没指定就读 prepare_data.py 写的 summary.json。
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

# --instance_prompt 是上游 argparse 的必填项，但用不到：每一行都有自己的 caption。
# 故意不开 --random_flip：翻转会改变旋转任务的语义。
# batch 必须是 1：四条指令长度 / image-pad 位置不同，训练脚本拒绝把不同 prompt 布局拼成一个 batch。
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
