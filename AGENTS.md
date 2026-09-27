# AGENTS.md

Shared rules for any coding agent working in this repository.

## Layout

- `src/qie_mnist/`: task data (`data.py`: tasks, `make_pairs(..., tasks=, next_target=)`, `next_prototypes()`), scoring
  (`evaluate.py`), probe loss (`probe.py`), plot style (`plotting.py`).
- `scripts/`: one entry point per step, run from the repo root. Each script's docstring names its step and the deck
  pages it serves; the README table "讲义页 → 脚本 → 结果文件" is the full map (keep it in sync when adding figures).
  - pipeline: `prepare_data.py` (`--tasks`, `--next-target`) → `train_lora.sh` (`--train-shift`, `--smoke`, ...)
    → `infer.py` (`edit`, `sweep --phase 1|2|3`) → `summarize_results.py`
  - analysis: `trace_forward.py` (A), `probe_loss.py` (D); figures: `plot_*.py` (default output `outputs/figures/`,
    file names match `deck/media/`).
- `train/`: vendored diffusers trainer (@ e0abab8) + `train_hooks.py`. Changes to the upstream file: hooks 1-4 and
  patch 5 (`--train_shift`). Keep the Apache-2.0 header and `NOTICE` in sync with any change, and regenerate
  `train/hooks.patch` against the pristine upstream file after editing the trainer.
- `results/`: small, path-free JSON summaries of the reported runs: `train/` (B), `train_e_shift5/` (E),
  `train_f_proto/` (F), `sweep/phase{1,2,3}/` (C), `probe/` (D). No images; the deck's figures live in `deck/media/`.
- `figures/`: the two README images that are not in the deck.
- `deck/`: Chinese lecture deck. Edit `deck/src/*.html`, then run `deck/build.sh`; commit both the sources and the built
  `deck/practice.html`. `deck/media/` holds only the figures the deck references.

## How to run

- Install: `pip install -r requirements.txt && pip install -e .`
- CPU checks (no GPU, no weights):
  `python -m py_compile src/qie_mnist/*.py scripts/*.py train/*.py`,
  `bash -n scripts/train_lora.sh`,
  `python scripts/prepare_data.py --out /tmp/qie_smoke --n-per-task 4`,
  `python scripts/summarize_results.py`,
  `python scripts/plot_train.py && python scripts/plot_probe.py && python scripts/plot_sweep.py --phase 3`.
- Everything that loads Qwen-Image 2.1 needs one CUDA GPU with >= 40 GB (runs used 1x H100 80GB).
- Experiment G (train on a task subset) is supported as an option (`--tasks`) but its results are not reported here.

## Hard rules

- **No weights in git.** Never commit Qwen-Image weights, LoRA checkpoints (`*.safetensors`), accelerate states or
  any other model files. The only allowed binary model file is `src/qie_mnist/assets/mnist_cls.pt` (~1.6 MB,
  self-trained). The Qwen weights are under the qwen-research license and must not be redistributed.
- **No secrets.** No API keys, tokens, credentials, private hostnames, bucket paths, personal paths or emails in
  code, logs, results or commit messages. Paths are CLI arguments; outputs go under gitignored `outputs/`.
- Do not commit large run dumps; only small, path-free JSON summaries belong in `results/`.
