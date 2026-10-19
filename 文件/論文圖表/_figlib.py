# -*- coding: utf-8 -*-
"""論文第三章圖表共用繪圖函式庫（matplotlib → 向量 PDF + PNG 預覽）。

設計重點：
- 座標單位＝英吋（圖寬即實際版面寬，建議 6.3in ≈ A4 內文寬），字級才與論文內文相稱
- `box()` 會自動縮小字級直到文字放得進框內（避免溢出），縮到 min 以下會警告
- PDF 用 TrueType(Type 42) 嵌入字型，LaTeX `\\includegraphics` 可直接使用
- 樣式（對應論文圖例）：solid＝已實作且第四章有證據；dashed＝已實作或已設計、無獨立驗證；
  gray＝設計但未實作

用法：
    cd 文件/論文圖表
    ../../.venv/Scripts/python.exe make_all.py
"""
from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "Noto Sans TC", "sans-serif"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 色票（低飽和、列印友善）
C = {
    "fcv": "#E4EEF9",      # ① FCV
    "res": "#E5F3E6",      # ② 研究部
    "app": "#FDF0DC",      # ③ 實戰層
    "llm": "#EFE7F6",      # LLM 模型層
    "db": "#EDEDED",
    "ink": "#222222",
    "mute": "#666666",
    "arrow": "#333333",
    "hl": "#C0392B",
}


class Canvas:
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self._renderer = self.fig.canvas.get_renderer()
        self.warnings: list[str] = []

    # ---------- 文字：自動縮字 ----------
    def _fit(self, text: str, w: float, h: float, fs: float, minfs: float,
             weight: str, color: str, ha: str = "center", x: float | None = None,
             y: float | None = None):
        pad = 0.035
        t = self.ax.text(0, 0, text, fontsize=fs, ha=ha, va="center", color=color,
                         fontweight=weight, linespacing=1.18, zorder=5)
        size = fs
        while True:
            bb = t.get_window_extent(self._renderer)
            tw, th = bb.width / self.fig.dpi, bb.height / self.fig.dpi
            if (tw <= w - 2 * pad and th <= h - 2 * pad) or size <= minfs:
                break
            size -= 0.2
            t.set_fontsize(size)
        if size < minfs + 1e-6 and not (tw <= w - 2 * pad and th <= h - 2 * pad):
            self.warnings.append(f"文字放不進框（{w:.2f}x{h:.2f}in，字級已縮到{size:.1f}）：{text[:24]!r}")
        return t, size

    def box(self, x, y, w, h, text, fs=7.0, style="solid", fc="#FFFFFF", bold=False,
            minfs=5.0, rounding=0.05, lw=0.9, color=None, ha="center"):
        """x,y＝左下角。style: solid / dashed / gray / plain。"""
        ec, ls, tcol = C["ink"], "-", color or C["ink"]
        if style == "dashed":
            ls = (0, (4, 2))
        elif style == "gray":
            ec, ls, tcol, fc = "#8A8A8A", (0, (1, 1.6)), color or "#6B6B6B", "#EFEFEF"
        elif style == "infra":
            ec, fc, lw = "#6B6B6B", "#F3F3F3", 0.7
        elif style == "plain":
            ec, lw = "none", 0
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                           fc=fc, ec=ec, lw=lw, ls=ls, zorder=2)
        self.ax.add_patch(p)
        tx = x + w / 2 if ha == "center" else x + 0.05
        t, _ = self._fit(text, w, h, fs, minfs, "bold" if bold else "normal", tcol,
                         ha=ha)
        t.set_position((tx, y + h / 2))
        return p

    def group(self, x, y, w, h, title, fc="#F7F7F7", fs=8.0, title_h=0.22, bold=True,
              ec=None, ls="-"):
        """外框＋上方標題列。"""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                         fc=fc, ec=ec or C["ink"], lw=1.1, ls=ls, zorder=1))
        if title:
            t, _ = self._fit(title, w - 0.1, title_h, fs, 5.5, "bold" if bold else "normal",
                             C["ink"])
            t.set_position((x + w / 2, y + h - title_h / 2 - 0.03))

    def badge(self, x, y, text, fs=6.4, w=None, fc="#2F4B7C"):
        """步驟標籤（深色底白字小膠囊），x,y 為中心點，通常壓在框的左上緣。"""
        w = w or (0.20 + 0.075 * len(text))
        h = 0.17
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                         boxstyle="round,pad=0,rounding_size=0.07", fc=fc,
                                         ec="white", lw=0.8, zorder=7))
        self.ax.text(x, y, text, fontsize=fs, color="white", ha="center", va="center",
                     fontweight="bold", zorder=8)

    def label(self, x, y, text, fs=6.5, color=None, ha="center", va="center", bold=False,
              bg=True, rot=0):
        self.ax.text(x, y, text, fontsize=fs, ha=ha, va=va, color=color or C["ink"],
                     fontweight="bold" if bold else "normal", rotation=rot, zorder=6,
                     linespacing=1.15,
                     bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.95) if bg else None)

    def arrow(self, x1, y1, x2, y2, label=None, fs=6.0, lw=1.0, color=None, style="-|>",
              ls="-", rad=0.0, label_pos=0.5, label_dx=0.0, label_dy=0.0, lcolor=None):
        self.ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=4,
                         arrowprops=dict(arrowstyle=style, lw=lw, color=color or C["arrow"],
                                         ls=ls, shrinkA=0, shrinkB=0, mutation_scale=7,
                                         connectionstyle=f"arc3,rad={rad}"))
        if label:
            lx = x1 + (x2 - x1) * label_pos + label_dx
            ly = y1 + (y2 - y1) * label_pos + label_dy
            self.label(lx, ly, label, fs=fs, color=lcolor)

    def line(self, pts, lw=1.0, color=None, ls="-", arrow_end=True):
        xs, ys = zip(*pts)
        if arrow_end:
            self.ax.plot(xs[:-1], ys[:-1], lw=lw, color=color or C["arrow"], ls=ls, zorder=4,
                         solid_capstyle="butt")
            self.arrow(xs[-2], ys[-2], xs[-1], ys[-1], lw=lw, color=color, ls=ls)
        else:
            self.ax.plot(xs, ys, lw=lw, color=color or C["arrow"], ls=ls, zorder=4)

    def cylinder(self, x, y, w, h, text, fs=6.5, fc=None):
        """資料庫圓柱圖示：x,y 左下角。"""
        fc = fc or C["db"]
        e = h * 0.22
        self.ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=fc, ec=C["ink"], lw=0.9, zorder=2))
        self.ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=fc, ec=C["ink"], lw=0.9, zorder=2))
        self.ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc="#FFFFFF", ec=C["ink"],
                                  lw=0.9, zorder=3))
        t, _ = self._fit(text, w - 0.04, h - e * 1.4, fs, 5.0, "normal", C["ink"])
        t.set_position((x + w / 2, y + (h - e) / 2 + e / 2))

    def legend(self, x, y, fs=6.0, vertical=False, gap=1.5):
        """圖例：三種框樣式。"""
        items = [("solid", "已實作，第四章有證據"), ("dashed", "已實作，無獨立驗證"),
                 ("gray", "設計但未實作"), ("infra", "基礎設施（非驗證對象）")]
        for i, (st, txt) in enumerate(items):
            dx, dy = (0, -0.2 * i) if vertical else (i * gap, 0)
            self.box(x + dx, y + dy, 0.28, 0.13, "", style=st, rounding=0.025)
            self.ax.text(x + dx + 0.34, y + dy + 0.065, txt, fontsize=fs, va="center",
                         color=C["ink"], zorder=6)

    def save(self, name: str, dpi: int = 220):
        pdf = os.path.join(OUT_DIR, f"{name}.pdf")
        png = os.path.join(OUT_DIR, f"{name}.png")
        self.fig.savefig(pdf)
        self.fig.savefig(png, dpi=dpi)
        plt.close(self.fig)
        for w in self.warnings:
            warnings.warn(f"[{name}] {w}")
        return pdf, png
