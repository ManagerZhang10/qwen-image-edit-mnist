# 讲解 deck

`practice.html` 是配合本仓库的中文讲解（19 页，单文件 HTML，图片在 `media/`）。Chrome 打开，
`←` / `→` 翻页，`G` 看页面列表，动画页按 `J` / `K` 走步。

主线：实验设置 → 训练流程 → 模型细节 → 训练 loss（按任务、按 σ 拆开）→ 生图效果与推理设置。

- `src/`：`head.html`、`stage.html`、`practice.html`（正文）、`tail.html`，改完跑 `./build.sh` 重新拼出 `practice.html`。
- `media/`：deck 用到的图，由本仓库的实验结果画出，不含任何模型权重。
