#!/bin/bash
# Build the lecture deck: concatenate the four parts under src/ into practice.html (single file, images in media/).
cd "$(dirname "$0")"
{ cat src/head.html src/stage.html src/practice.html src/tail.html; } > practice.html
grep -c '<section class="slide' practice.html
