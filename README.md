# qwen-image-edit-mnist

**English summary.** A small, fully reproducible practice project on the **Qwen-Image 2.1** image-editing model
(~16B parameters), built on four synthetic MNIST edits: rotate 90°, rotate 180°, "next digit" and invert. It walks
through one real model end to end: (A) a tensor-level trace of one edit forward pass, (B) LoRA fine-tuning with the
official diffusers img2img trainer (vendored, Apache-2.0, plus logging hooks), (C) inference sweeps over CFG /
shift / steps / LoRA scale, (D) loss vs noise level σ and the σ = 1 floor, and two training variants, (E) train-time
σ shift and (F) fixed "next" targets. A rank-32 LoRA trained for 3000 steps on 2000 pairs (1× H100, ~1.4 GPU-hours)
learns the deterministic edits (rotations 100 %, invert 94 %, IoU 0.94–0.99), and after fine-tuning they work in
2 sampling steps. "Next digit" stays at 6–31 %, and neither variant fixes it. Every script is tied to a page of the
Chinese lecture deck in `deck/`. **No weights are included**: the base model is under the qwen-research license and
the trained LoRA is not released. The rest of this README is in Chinese.

---

## 这个项目教什么

用一个小到能完全看清楚的任务，把一个真实的指令编辑模型从头到尾走一遍：数据怎么造、一个训练 step 在算什么、
张量在模型里怎么走、训练 loss 按任务和 σ 拆开后在说什么、成功率怎么打分、推理时 CFG / shift / 步数 / LoRA 强度
各管什么。四个任务里三个答案唯一（旋转、反色），一个答案不唯一（「下一个数字」的目标是另一个人写的 n+1），
学得会和学不会的对比贯穿全程。

配套讲义是 [`deck/practice.html`](deck/practice.html)（19 页中文幻灯片，下载后用 Chrome 打开，`←` / `→` 翻页，
说明见 [`deck/README.md`](deck/README.md)）。下文的「[讲义页 → 脚本 → 结果文件](#讲义页--脚本--结果文件)」表把每一页
对到仓库里的脚本和数据，建议打开讲义，一页一页对着代码看。

| 实验 | 内容 | 入口 |
| --- | --- | --- |
| A 前向追踪 | 一次真实编辑前向的每一步：预处理、VAE、token 打包、文本编码、联合序列、块因果 mask、RoPE、调制、KV 缓存、Euler 更新；手写镜像循环与真实 `pipe(...)` 逐像素一致 | `scripts/trace_forward.py` |
| B LoRA 训练 | diffusers 官方 img2img LoRA 脚本，只加只读挂钩：每步 σ、固定测试集 probe loss、定期验证 | `scripts/train_lora.sh` |
| C 推理扫描 | 第 1 轮底模，第 2 轮 LoRA 强度，第 3 轮带 LoRA 的 CFG / shift / 步数 + 每步 x0′ 轨迹 | `scripts/infer.py sweep` |
| D loss 与 σ | 每个任务的 loss vs σ、σ = 1 处的下界 D = E_c[Var(x0 \| c)]、v loss = x0 误差 × 1/σ² | `scripts/probe_loss.py` |
| E 训练时 σ shift | 其余同 B，只把训练 σ 映射成 5σ/(1+4σ)，三分之一的步落在 σ > 0.9 | `train_lora.sh --train-shift 5` |
| F next 固定原型 | 其余同 B，next 的目标改成每个数字一张固定原型（答案变唯一） | `prepare_data.py --next-target proto` |

![四个任务](deck/media/qi21_tasks.png)

## 四个任务和固定指令

| 任务 | 指令（固定中文） | 目标 | 答案唯一？ |
| --- | --- | --- | --- |
| `rot90` | 把图片顺时针旋转 90 度 | 参考图顺时针转 90° | 是 |
| `rot180` | 把图片旋转 180 度 | 参考图转 180° | 是 |
| `next` | 把数字换成下一个数字，9 变成 0 | **另一个人写的** n+1（按种子从同一 split 里抽）；`--next-target proto` 时为固定原型 | 否（proto 时是） |
| `invert` | 把图片黑白反色 | 255 − 参考图 | 是 |

28×28 灰度图用双三次插值放大到 512×512 RGB。训练集 `make_pairs('train', 500)` 共 2000 对，测试集
`make_pairs('test', 16)` 共 64 对（**每个任务 16 张**）。评测时把输出缩回 28×28，先做逆变换（转回来、反色回来），
再用自训练的小 CNN（`src/qie_mnist/assets/mnist_cls.pt`，1.6 MB，MNIST 测试集准确率 98.8%）分类，认出的数字等于
期望数字就算成功。确定性任务另外报告和标准答案的像素 IoU。**每个任务只有 16 张，一张图就是 6.25 个百分点**，
几个点的差异都在噪声范围内。

## 目录结构

```text
src/qie_mnist/        data.py      任务、固定指令、样本对生成（next 的 random / proto 两种目标、任务子集）
                      evaluate.py  分类器、逆变换、成功率、IoU
                      probe.py     固定 σ / 固定噪声的 flow-matching loss（训练挂钩和实验 D 共用）
                      plotting.py  画图风格（白底、单一强调色、中文字体）
                      assets/mnist_cls.pt  自训练分类器（scripts/train_classifier.py 可重新生成）
scripts/              prepare_data.py      第 1 步 数据
                      train_lora.sh        第 2 步 训练（B / E / F / 只训部分任务）
                      infer.py             第 3 步 单张推理 / 推理设置扫描（C 第 1–3 轮）
                      summarize_results.py 第 4 步 把结果 JSON 打印成下文的结论表
                      trace_forward.py     分析 A：一次前向的张量追踪
                      probe_loss.py        分析 D：loss vs σ、σ 扫描可视化
                      plot_*.py            第 5 步 画讲义里的图（见对照表）
                      train_classifier.py  重新训练评测用分类器
train/                train_dreambooth_lora_qwenimage21_img2img.py  diffusers @ e0abab8（Apache-2.0）+ 4 个挂钩 + --train_shift
                      train_hooks.py  挂钩实现        hooks.patch  相对上游文件的完整 diff
results/              train/             B 的 val / probe / train 日志和 σ 汇总
                      train_e_shift5/    E 的 val / probe 日志和 σ 汇总
                      train_f_proto/     F 的 val / probe 日志
                      sweep/phase{1,2,3}/  C 各轮的 metrics（phase3 另有 x0trace.json）
                      probe/             D 的 results.json、vis_numbers.json
deck/                 讲义（practice.html + media/ 里的图）
figures/              README 里另外用到的两张图
```

## 硬件要求

| 环节 | 硬件 | 显存峰值 | 耗时 |
| --- | --- | --- | --- |
| 数据、评测汇总、大部分画图、分类器训练 | CPU | — | 秒到分钟级 |
| B 训练（3000 步 + 16 次验证 + 每 50 步 probe） | 1× H100 80GB | 36 GiB | 训练约 0.24–0.35 s/步；每轮验证（64 张 × 40 步）约 150 s；整个作业 1.5–1.7 h |
| E / F 训练 | E：1× RTX PRO 6000 Blackwell 96 GB；F：1× H100 80GB | 36 GiB | 与 B 同量级（F 1.6 h，E 2.3 h） |
| A 前向追踪 | 1× H100 80GB | 33 GiB | 约 30 s（不含下载权重）；采样每步 38 ms（KV 缓存开）/ 73 ms（关） |
| C 第 1、2 轮扫描 | 1× 96 GB 卡（RTX PRO 6000 Blackwell） | — | 第 1 轮 14 组约 44 min，第 2 轮 7 组约 21 min |
| C 第 3 轮扫描 | 1× H100 80GB | — | 11 组 + x0′ 轨迹约 28 min |
| D probe | 1× 96 GB 卡 | — | 约 13 min |

任何加载 Qwen-Image 2.1 的脚本都需要一张 **≥ 40 GB 显存的 CUDA 卡**（bf16，全部组件放在 GPU 上）。

## 安装

```bash
git clone https://github.com/ManagerZhang10/qwen-image-edit-mnist.git && cd qwen-image-edit-mnist
python -m venv .venv && source .venv/bin/activate      # Python 3.10+，实验用 3.12
# 先按 CUDA 版本装 torch，例如 CUDA 12.8：
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

只在 CPU 上看数据、画图时，`pip install torch torchvision numpy pillow matplotlib pyarrow && pip install -e . --no-deps` 就够了。

**diffusers 必须是 commit `e0abab83b5df05de9e7abd788643c1a7c1e42e28`**：`QwenImage21Pipeline`、
`AutoencoderKLQwenImage21` 和 `train/` 下的训练脚本都依赖这个版本，`requirements.txt` 已固定为该 commit 的源码包。

## 下载基础模型权重（约 31 GiB）

权重来自 Hugging Face 上的 [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)，本实验固定在
revision **`790c92633540aa0cb11d9abf19eb46d861714758`**（diffusers 格式快照）。

> **许可证：Qwen-Image 2.1 的权重使用 qwen-research license，不是 Apache-2.0。**
> 下载前请到模型页自行阅读并接受该许可，使用范围以许可条款为准。**本仓库不包含、也不分发任何权重**，
> 本仓库的 Apache-2.0 只覆盖这里的代码。实验训练出的 **LoRA 也不发布**，需要按下文命令自己训练。

```bash
hf download Qwen/Qwen-Image-2.1 \
  --revision 790c92633540aa0cb11d9abf19eb46d861714758 \
  --local-dir models/Qwen-Image-2.1
# 旧版 CLI 等价写法：
# huggingface-cli download Qwen/Qwen-Image-2.1 --revision 790c92633540aa0cb11d9abf19eb46d861714758 --local-dir models/Qwen-Image-2.1
```

- 大小：约 31 GiB（33 GB）。`transformer/` 13.25 GiB，`text_encoder/`（Qwen3-VL）16.33 GiB，`vae/` 1.26 GiB。
- **tokenizer 和图像处理器在 `processor/` 目录下**（不是常见的 `tokenizer/`），pipeline 会自动读取。
- 写这份文档时该 revision 不需要登录即可下载；如果以后模型页改成需要同意许可，先 `hf auth login`。
- 下文的 `MODEL=models/Qwen-Image-2.1` 就指这个目录；只跑第 3 页缩略图时只需要其中的 `vae/`。

## 快速上手：一条命令一个环节

以下命令都在仓库根目录运行。**标 GPU 的需要上面的权重和一张 ≥ 40 GB 的卡**，其余只用 CPU。

```bash
MODEL=models/Qwen-Image-2.1
```

**0. CPU 自检**（不需要 GPU 和权重，几十秒）

```bash
python scripts/prepare_data.py --out outputs/smoke_data --n-per-task 4
python scripts/summarize_results.py        # 用仓库附带的 results/ 打印下文所有结论表
```

**1. 准备数据**（CPU，MNIST 自动下载到 `./data`）

```bash
python scripts/prepare_data.py --out data/mnist_edit_train                       # B / E：4 × 500 = 2000 对
python scripts/prepare_data.py --out data/mnist_edit_proto --next-target proto   # F：next 目标换成固定原型
python scripts/prepare_data.py --out data/mnist_edit_next --tasks next           # 只训部分任务（逗号分隔，如 rot90,invert）
```

`--next-target proto` 时每个任务抽到的参考图、以及其它三个任务的目标都与默认模式完全相同，只有 next 的目标变了。

**2. 训练 LoRA**（GPU，约 1.5 h / H100）

```bash
bash scripts/train_lora.sh $MODEL data/mnist_edit_train outputs/lora_smoke --smoke        # 6 步冒烟测试，先跑这个
bash scripts/train_lora.sh $MODEL data/mnist_edit_train outputs/lora_b                    # B：基线
bash scripts/train_lora.sh $MODEL data/mnist_edit_train outputs/lora_e --train-shift 5    # E：训练时 σ shift 5
bash scripts/train_lora.sh $MODEL data/mnist_edit_proto outputs/lora_f                    # F：自动读 summary.json，测试集 next 也用原型
bash scripts/train_lora.sh $MODEL data/mnist_edit_next  outputs/lora_next                 # 只训 next（测试集仍是四个任务）
# 产物：outputs/lora_b/ckpt-3000/pytorch_lora_weights.safetensors、outputs/lora_b/hook_logs/{train,probe,val}.jsonl
```

其它选项：`--steps`、`--val-every`、`--probe-every`、`--n-val`、`--resume latest`，见 `bash scripts/train_lora.sh --help`。

**3. 推理 / 推理设置扫描**（GPU）

```bash
python scripts/infer.py edit --model $MODEL --task rot90 --index 0 --out outputs/edit_base
python scripts/infer.py edit --model $MODEL --task invert --lora outputs/lora_b/ckpt-3000 --steps 4 --out outputs/edit_lora
python scripts/infer.py sweep --model $MODEL --phase 1 --out outputs/sweep/phase1                                 # 底模的旋钮
python scripts/infer.py sweep --model $MODEL --phase 2 --lora outputs/lora_b/ckpt-3000 --out outputs/sweep/phase2  # LoRA 强度
python scripts/infer.py sweep --model $MODEL --phase 3 --lora outputs/lora_b/ckpt-3000 --out outputs/sweep/phase3  # 带 LoRA 的 CFG/shift/步数 + x0′ 轨迹
```

**4. 评测汇总**（CPU）

```bash
python scripts/summarize_results.py b --train outputs/lora_b/hook_logs          # 训练过程的成功率、首次 ≥ 90% 的步数
python scripts/summarize_results.py e f --e outputs/lora_e/hook_logs --f outputs/lora_f/hook_logs --train outputs/lora_b/hook_logs
python scripts/summarize_results.py c --sweep outputs/sweep                     # 推理设置对比
```

**5. 分析**（GPU）

```bash
python scripts/trace_forward.py --model $MODEL --out outputs/trace                                   # A
python scripts/probe_loss.py measure --model $MODEL --lora outputs/lora_b/ckpt-3000 --out outputs/probe   # D
python scripts/probe_loss.py vis --model $MODEL --lora outputs/lora_b/ckpt-3000 --out outputs/probe/vis
```

**6. 画图复现**（CPU；默认写到 `outputs/figures/`，文件名和 `deck/media/` 一致，可以直接对比）

```bash
python scripts/plot_tasks.py                                   # 第 1、2 页
python scripts/plot_train.py                                   # 第 8、9、13、15 页（读 results/train/）
python scripts/plot_probe.py                                   # 第 11 页（读 results/probe/）
python scripts/plot_sweep.py --phase 2 && python scripts/plot_sweep.py --phase 3   # 第 16–19 页的曲线和柱状图（读 results/sweep/）
python scripts/plot_trace.py mask_simple                       # 第 6 页（示意图，不需要数据）
# 下面这些需要你自己跑出的图像输出（仓库只放小 JSON，不放图像）：
python scripts/plot_train.py --src outputs/lora_b/hook_logs    # 另加第 14 页（验证图）
python scripts/plot_probe.py --vis outputs/probe/vis           # 另加第 10 页
python scripts/plot_scoring.py --src outputs/sweep/phase3      # 第 12 页
python scripts/plot_sweep.py --phase 3 --src outputs/sweep     # 第 16、17、19 页带缩略图的完整版（第 2 轮同理）
python scripts/plot_trace.py --src outputs/trace prep flow     # 第 5 页；flow 是第 4 页内容的 matplotlib 版
python scripts/plot_train_step.py --vae $MODEL/vae --vis outputs/probe/vis   # 第 3 页的潜变量缩略图（需要 VAE 和 diffusers）
```

没有 GPU 输出时，`plot_sweep.py` 仍会画出完整版式，缩略图的位置换成一行提示；完整的图直接看 `deck/media/`。

## 讲义页 → 脚本 → 结果文件

页码按 `deck/practice.html` 的页码（封面 = 1）。「仓库数据」一栏写明能否只用 `results/` 重画；「需要输出」表示需要你自己
跑出的图像输出（仓库不放大文件，讲义里的成图在 `deck/media/`）。

| 页 | 标题 | 产生数据的脚本 | 画图脚本 → 图 | 仓库数据 |
| --- | --- | --- | --- | --- |
| 1 | 封面 | `src/qie_mnist/data.py` | `plot_tasks.py` → `practice_strip.png` | 不需要数据（CPU 现生成） |
| 2 | 四种编辑任务 | `data.py`、`prepare_data.py` | `plot_tasks.py` → `qi21_tasks.png` | 不需要数据 |
| 3 | 训练一步怎么走 | `probe_loss.py vis`（x0′ 缩略图）+ 权重里的 VAE | `plot_train_step.py` → `tf_*.png` | 需要 VAE 和 `outputs/probe/vis/rot90_11_x0hat_0.5.png` |
| 4 | 张量怎么走 | `trace_forward.py` | **静态图（draw.io），不提供生成脚本**；`plot_trace.py flow` 用同一份追踪数据画 matplotlib 版 | 需要输出（trace） |
| 5 | 参考图的两条预处理 | `trace_forward.py` | `plot_trace.py prep` → `qi21_a_prep.png` | 需要输出（trace） |
| 6 | 块因果注意力 | `trace_forward.py`（扰动检查的数字） | `plot_trace.py mask_simple` → `qi21_a_mask_simple.png` | 不需要数据（示意图） |
| 7 | LoRA 挂在哪 | `train_lora.sh`（`--lora_layers`、rank 32） | **静态图（私有画图工具 + draw.io），不提供生成脚本** | — |
| 8 | 训练 loss | `train_lora.sh` → `hook_logs/train.jsonl`、`probe.jsonl` | `plot_train.py` → `qi21_b_loss.png` | `results/train/` ✓ |
| 9 | 按 σ 看 loss | `train_lora.sh` → `probe.jsonl` | `plot_train.py` → `qi21_b_probe_by_task.png` | `results/train/` ✓ |
| 10 | σ 扫描看 x0 猜测 | `probe_loss.py vis` | `plot_probe.py --vis` → `qi21_b_sigma_vis.png` | 数字 `results/probe/vis_numbers.json`；缩略图需要输出 |
| 11 | loss = x0 误差 × 1/σ² | `probe_loss.py measure` | `plot_probe.py` → `qi21_b_sigma_decomp.png` | `results/probe/results.json` ✓ |
| 12 | 成功率怎么算 | `infer.py sweep --phase 3`（lora1.0 的输出）、`evaluate.py` | `plot_scoring.py` → `qi21_c_scoring.png` | 需要输出（5 张生成图） |
| 13 | 成功率 | `train_lora.sh` → `val.jsonl` | `plot_train.py` → `qi21_b_success.png` | `results/train/` ✓ |
| 14 | 训练前后对照 | `train_lora.sh` → `hook_logs/val/step_*/` | `plot_train.py --src <hook_logs>` → `qi21_b_samples.png` | 需要输出（验证图） |
| 15 | 学得会 vs 学不会 | `train_lora.sh` → `val.jsonl`、`probe.jsonl` | `plot_train.py` → `qi21_b_task_order.png` | `results/train/` ✓ |
| 16 | CFG | `infer.py sweep --phase 3` | `plot_sweep.py --phase 3` → `qi21_c_lora_cfg.png` | 数字 `results/sweep/phase3/` ✓；缩略图需要输出 |
| 17 | shift | `infer.py sweep --phase 3` | `plot_sweep.py --phase 3` → `qi21_c_lora_shift.png` | 同上 |
| 18 | LoRA 强度 | `infer.py sweep --phase 2` | `plot_sweep.py --phase 2` → `qi21_c_lora_scale.png` | 数字 `results/sweep/phase2/` ✓；缩略图需要输出 |
| 19 | 采样步数 | `infer.py sweep --phase 3` | `plot_sweep.py --phase 3` → `qi21_c_lora_steps.png`（另出 `qi21_c_lora_x0hat.png`） | 数字 `results/sweep/phase3/` ✓；缩略图需要输出 |

## 主要结论

完整数字用 `python scripts/summarize_results.py` 从 `results/` 打印。**每个任务 16 张测试图，一张 = 6.25 个百分点**，
下面几个点的差别都可能只是一两张图。

### B 训练前后（40 步采样，CFG 1）

| | 顺时针 90° | 180° | 下一个数字 | 反色 | 平均 |
| --- | --- | --- | --- | --- | --- |
| 底模（第 0 步） | 75%（IoU 0.37） | 62%（0.44） | 25% | 38%（0.01） | 50.0% |
| 第 200 步 | 81%（0.40） | **100%（0.97）** | 6% | 19%（0.00） | 51.6% |
| LoRA 第 3000 步 | **100%（0.94）** | **100%（0.97）** | 12% | **94%（0.99）** | 76.6% |
| 首次 ≥ 90% | 第 400 步 | 第 200 步 | 3000 步内未达到 | 第 600 步 | |

- **学得会的**：两种旋转和反色是逐像素的确定性变换，几百步就学会，IoU 0.94–0.99。底模会旋转但会把数字重画（IoU 0.37），
  从来不做反色（IoU 0.01，那 38% 是分类器在未反色的原图上碰巧判对）。
- **学不会的**：「下一个数字」的目标是另一个人写的 n+1，同一个输入没有唯一答案，3000 步里成功率一直在 0–19%。
- **loss 和成功率脱节**：固定测试集 loss 第 200 步就完成了全程降幅的约八成（0.043 → 0.025 → 最终 0.020），
  成功率却在 200–600 步才台阶式跳上去；第 200 步反色还先从 38% 掉到 19%。

### C 推理设置（第 3000 步 LoRA，同一批 64 个输入和噪声）

| 旋钮（第 3 轮，LoRA 强度 1.0） | 平均成功率 | 下一个数字 | 其余三个任务 | 每张耗时 |
| --- | --- | --- | --- | --- |
| CFG 1（默认）/ 2 / 4 / 7 | 75.0% / 78.1% / 81.2% / 81.2% | 6% / 19% / 31% / 31% | 不变（100 / 100 / 94%） | 2.3 s → 4.6 s（CFG > 1 每步两次前向） |
| shift 固定 1 / 默认动态（≈1.71）/ 3 / 6，40 步 | 75.0% / 75.0% / 76.6% / 76.6% | 6% / 6% / 12% / 12% | 不变 | 相同 |
| 采样 2 / 4 / 8 / 16 / 40 步 | 76.6% / 75.0% / 75.0% / 78.1% / 75.0% | 12% / 6% / 6% / 19% / 6% | 2 步起就是 100 / 100 / 94% | 0.29 s（2 步）对 2.33 s（40 步），快 8 倍 |

| 旋钮（第 2 轮） | 平均成功率 | 下一个数字 | 说明 |
| --- | --- | --- | --- |
| LoRA 强度 0 / 0.5 / 1.0 / 1.5 | 46.9% / **81.2%** / 76.6% / 73.4% | 25% / 31% / 12% / 0% | 强度 0 与底模逐像素相同；0.5 已学会旋转和反色、彩色伪影消失；> 1 继续压低「下一个数字」 |

- **结论**：微调后答案唯一的编辑 **2 步就够**；CFG 只帮「下一个数字」（代价是时间翻倍）；40 步下 shift 几乎没影响
  （少步 × 不同 shift 没测）；LoRA 强度 0.5 就够用。「下一个数字」少步时是一团模糊的「平均图」，16 步起才画成具体数字。
- 第 3 轮在 H100 上跑，第 1、2 轮在 RTX PRO 6000 上：同一配置（LoRA 1.0、40 步）第 3 轮 75.0%、第 2 轮 76.6%，
  差在「下一个数字」的一张图；耗时也不能跨轮比较。
- 第 1 轮（底模）的发现：CFG 4 / 7 让旋转更听话（平均 46.9% → 57.8% / 60.9%）；2 步输出又暗又糊；KV 缓存提速 1.91×
  且 64 个判定完全一致；关掉 causal_condition 画面反而更干净（**只是这个小任务上的观察，没有和官方推理代码核对**）。

### D loss 与 σ（LoRA 第 3000 步，16 张 × 4 组噪声的平均）

| σ | 0.1 | 0.5 | 0.9 | 0.99 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| 90° / 180° / 反色 | 0.056 / 0.042 / 0.043 | 0.010 / 0.004 / 0.005 | 0.006 / 0.002 / 0.003 | 0.009 / 0.003 / 0.005 | 0.023 / 0.017 / 0.031 |
| 下一个数字 | 0.057 | 0.018 | 0.060 | 0.513 | 0.690 |

- σ = 1 时 x_t 不含 x0 的信息，最优预测是 E[x0|c]，loss 的下界是 **D = E_c[Var(x0|c)]**。确定性任务 D = 0；
  「下一个数字」在训练集 latent 里实测 **D ≈ 0.36**。
- **恒等式 v loss = x0 误差 × 1/σ²**：左头（σ 小）是 1/σ² 放大，所有任务都有；右头（σ → 1）是看不清目标、x0 误差猛涨，
  只有答案不唯一的「下一个数字」明显，所以它的曲线是 U 形。
- 「下一个数字」σ = 0.9 时 loss 只有 D 的 1/6（图像结构让 x_t 仍泄露大部分 x0），σ = 1 时为 0.69 ≈ 1.9 × D：
  多出的是偏差项，模型没学会这个任务。

![loss vs sigma](figures/qi21_b_loss_vs_sigma_by_task.png)

### E / F：两个训练变体（其余设置都同 B）

| | B（基线） | E（训练时 σ shift 5） | F（next 用固定原型） |
| --- | --- | --- | --- |
| 训练时实际 σ：平均 / σ > 0.9 占比 | 0.50 / 10% | 0.75 / 34% | 同 B |
| 首次 ≥ 90%：90° / 180° / 反色 | 400 / 200 / 600 步 | **200 / 200 / 200 步** | 800 / 200 / 600 步 |
| 第 200 步平均成功率 | 51.6% | **76.6%**（反色 19% → 94%） | 51.6% |
| 第 3000 步平均成功率 | 76.6% | 76.6% | 78.1% |
| next 成功率（第 200–3000 步平均 / 最高） | 7.5% / 19% | 7.5% / 19% | 6.7% / 19% |
| next 抄原数字比例（同上平均） | 0.16 | 0.08 | 0.10 |
| 第 3000 步 probe loss，σ = 0.1 | 基准 | 各任务高 11–18%（小 σ 练得少） | 三个确定性任务持平 |
| 第 3000 步 probe loss，next σ = 0.9 | 0.060 | 0.061 | 0.046（目标换了，绝对值不能直接比） |

- **E**：把三分之一的训练步放到 σ > 0.9，确定性编辑按步数收敛快 2–3 倍，但最终数字与 B 相同，「下一个数字」照样学不会。
- **F**：目标固定后 next 的 loss 确实降了，但成功率、IoU（对原型始终 0.09–0.23）都和 B 在同一噪声区间；同一个原型目标的
  4 个「1→2」测试样本被画成 9 / 0 / 9 / 3。另外 F 的 90° 在第 400–600 步有一段下跌（到 31%），第 800 步恢复，原因没查。
- 两者合起来：「下一个数字」学不会，既不是高 σ 练得不够，也不只是目标方差（D）造成的。
- E 的第 0 步（底模）验证与 B 差一张图（90° 69% vs 75%）：B 和 F 跑在 H100 上，第 0 步的预测完全一致；E 跑在
  RTX PRO 6000 Blackwell 上，64 张里有 2 张预测不同，其中 rot90 一张从成功变成失败。不同 GPU 型号的 bf16 数值差异
  会让临界样本翻转，同型号 GPU 上可复现。

### A 前向追踪的几个事实

零样本 rot90，第 5 步（σ = 0.94）的一步预测 x0′ 已经是转好的数字，后面 35 步只在修细节：

![x0 hat](figures/qi21_a_x0hat.png)

- 512² 的图经 16× VAE 变成 32×32×64，一个 token = 16×16 像素，**不做 2×2 打包**。
- 联合序列 2073 个 token：文本 8 → 参考图 1024 → 指令等 17 → 目标 1024。块因果 mask：文本段下三角，每张图内部全连通，
  所以**参考图 token 看不到后面的指令**（扰动实测差 0.0）。
- 前缀 1049 个 token 用 t = 0 的调制行，与步数无关，可以只算一次放进 KV 缓存；之后每步只算 1024 个目标 token，采样快约 1.9×。
- 文本特征取 Qwen3-VL 最后一层**过 final norm 之前**的输出（std 11.2 vs 2.3）。transformers 5.x 默认返回过 norm 的值，pipeline 用 hook 抵消。

## 训练超参（实验 B，`scripts/train_lora.sh` 默认值）

| 项 | 值 |
| --- | --- |
| 数据 | 2000 对（4 任务 × 500），batch 1（不同指令的 image-pad 位置不同，脚本不允许混 batch），3000 步 = 1.5 epoch |
| LoRA | rank 32，alpha 32，层 `to_k,to_q,to_v,to_out.0,img_mlp.proj,img_mlp.gate_layer,img_mlp.out`，448 个张量，83.9 M 参数 |
| 优化 | AdamW，lr 1e-4 constant，无 warmup，weight decay 1e-4，梯度裁剪 1.0，seed 0 |
| 精度 / 分辨率 | bf16，512×512，`--cache_latents`，不开 `--random_flip`（翻转会改变旋转任务的语义） |
| σ 采样 | `--weighting_scheme none`：训练时 σ ~ U(0,1)，loss 权重 1；`--train-shift S` 时再映射成 Sσ/(1+(S−1)σ)（E 用 5）；采样时的 shift 与此无关 |
| 验证 | `make_pairs('test', 16)` 共 64 张，第 i 张种子 1000+i，40 步，CFG 1.0，每 200 步 |
| probe | 同样 64 张，σ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}，每张固定噪声（种子 1234+i），每 50 步 |

## 局限

- **指令是固定模板**：每个任务 1 句中文，没有改写。训练集 2000 对只用了 4 句话，模型学到的是「这 4 句话 → 这 4 种变换」，
  不能说明它能听懂换一种说法的指令。真实编辑数据需要人写或 LLM 扩写的多样指令。
- **样本量小**：每个任务 16 张测试图，一张 = 6.25 个百分点；C 第 1 轮的分辨率对比只用了 8 张。
- **评测靠一个 MNIST 分类器**：对「重画」宽容，对伪影敏感，对底模偏乐观；IoU 更能说明像素是否对得上。
- **「下一个数字」的失败**是这组数据量、LoRA rank 和 3000 步下的结果（E、F 也没救回来），没有试更多数据、更长训练或更高 rank。
- **causal_condition 关闭后效果更好**这一观察没有和官方推理实现核对，不应当作结论。
- **环境差异**：A、B、E、F 用 torch 2.11 + CUDA 12.8（A、B、F 在 H100，E 在 RTX PRO 6000 Blackwell）；C、D 用
  torch 2.13 + CUDA 13.0，C 第 1、2 轮与第 3 轮又在不同 GPU 上。diffusers 始终是 e0abab8，transformers 始终是 5.17.0。
  bf16 下不同 GPU 型号的数值不会逐位一致，临界样本的判定可能翻转；同型号 GPU 上可复现。
- **讲义第 4、7 页的图是静态资源**（draw.io / 私有画图工具），仓库不提供生成脚本。
- **不发布任何权重**：Qwen 底模权重请按上文从 Hugging Face 下载；实验训练的 LoRA 不公开，需要自己训练。

## 许可证

代码使用 [Apache-2.0](LICENSE)。`train/train_dreambooth_lora_qwenimage21_img2img.py` 来自 Hugging Face diffusers
（Apache-2.0），来源 commit 和改动见 [NOTICE](NOTICE)。Qwen-Image 2.1 权重使用 qwen-research license，
不在本仓库的许可范围内。MNIST 在运行时通过 torchvision 下载。
