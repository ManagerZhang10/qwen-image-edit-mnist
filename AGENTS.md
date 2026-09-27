# AGENTS.md

Shared rules for any coding agent working in this repository.

## Layout

- `src/qie_mnist/`: task data (`data.py`), scoring (`evaluate.py`), probe loss (`probe.py`), plot style (`plotting.py`).
- `scripts/`: entry points (prepare data, train, infer, trace, probe, plots). Run from the repo root.
- `train/`: vendored diffusers trainer (@ e0abab8) + `train_hooks.py`. Keep the Apache-2.0 header and `NOTICE` in sync
  with any change; regenerate `train/hooks.patch` against the upstream file after editing the trainer.
- `results/`: small JSON summaries of the reported runs. `figures/`: images used by the README.

## How to run

- Install: `pip install -r requirements.txt && pip install -e .`
- CPU checks (no GPU, no weights):
  `python -m py_compile src/qie_mnist/*.py scripts/*.py train/*.py`,
  `bash -n scripts/train_lora.sh`,
  `python scripts/prepare_data.py --out /tmp/qie_smoke --n-per-task 4`,
  `python scripts/plot_tasks.py --out /tmp/qie_figs`.
- Everything that loads Qwen-Image 2.1 needs one CUDA GPU with >= 40 GB (runs used 1x H100 80GB).

## Hard rules

- **No weights in git.** Never commit Qwen-Image weights, LoRA checkpoints (`*.safetensors`), accelerate states or
  any other model files. The only allowed binary model file is `src/qie_mnist/assets/mnist_cls.pt` (~1.6 MB,
  self-trained). The Qwen weights are under the qwen-research license and must not be redistributed.
- **No secrets.** No API keys, tokens, credentials, private hostnames, bucket paths, personal paths or emails in
  code, logs, results or commit messages. Paths are CLI arguments; outputs go under gitignored `outputs/`.
- Do not commit large run dumps; only small, path-free JSON summaries belong in `results/`.
