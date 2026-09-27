#!/bin/bash
# 拼装讲解 deck：src/ 下四段按顺序拼成 practice.html（单文件，图片在 media/）
cd "$(dirname "$0")"
{ cat src/head.html src/stage.html src/practice.html src/tail.html; } > practice.html
grep -c '<section class="slide' practice.html
