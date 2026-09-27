"""Shared matplotlib style for the figure scripts: white background, one accent colour, CJK-capable fonts
(the figure labels are Chinese, matching deck/media)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#FFFFFF"
PANEL = "#F5F5F7"
TEXT = "#1D1D1F"
MUTED = "#6E6E73"
ACCENT = "#0071E3"
GRAY = "#86868B"
LIGHT = "#D2D2D7"

TASK_ZH = {"rot90": "顺时针转 90°", "rot180": "旋转 180°", "next": "换成下一个数字", "invert": "黑白反色"}

CJK_FONTS = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC",
             "Noto Sans CJK JP", "Noto Sans SC", "WenQuanYi Zen Hei"]
# macOS system font files that are not always registered with matplotlib by default.
_FONT_FILES = ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"]


def cjk_fonts():
    import logging
    from matplotlib import font_manager
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    for f in _FONT_FILES:
        if Path(f).exists():
            try:
                font_manager.fontManager.addfont(f)
            except Exception:  # noqa: BLE001
                pass
    have = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in CJK_FONTS if f in have]
    if not fonts:
        print("WARNING: no CJK font found (e.g. install Noto Sans CJK); Chinese labels will render as boxes",
              flush=True)
    return fonts


def set_style(font_size=18):
    plt.rcParams.update({
        "font.family": cjk_fonts() + ["DejaVu Sans", "sans-serif"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 3,
        "ytick.labelsize": font_size - 3,
        "legend.fontsize": font_size - 3,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    })


def save(fig, out_dir, name, **kw):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=kw.pop("dpi", 150), **kw)
    plt.close(fig)
    print("saved", p)
    return p
