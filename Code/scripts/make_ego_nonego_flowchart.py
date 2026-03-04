#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ego_nonego_flowchart.py
============================
Publication-quality flowchart of the rule-based Ego / Non-ego
classification method (v7).

Run:
    python3 scripts/make_ego_nonego_flowchart.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse

# ── paths ────────────────────────────────────────────────────────────
CODE = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver/Code")
OUT  = CODE / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────
EGO_CLR    = "#54A24B"
NONEGO_CLR = "#E45756"
BLUE       = "#4C78A8"
PROC_FC    = "#E8F0FE"
DEC_FC     = "#FFF8E1"
ARROW_CLR  = "#444444"
EDGE_CLR   = "#555555"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          8,
    "axes.linewidth":     0.0,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
})


# ── primitives ───────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, text, fc="#fff", ec=EDGE_CLR, fs=7,
         fw="normal", tc="black", lw=0.8):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.012", fc=fc, ec=ec,
                       lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=fw, color=tc, zorder=3, linespacing=1.22,
            clip_on=False)


def _diamond(ax, cx, cy, w, h, text, fc=DEC_FC, ec=EDGE_CLR,
             fs=6.5, lw=0.8):
    verts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2),
             (cx-w/2, cy), (cx, cy+h/2)]
    ax.add_patch(Polygon(verts, closed=True, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.18, clip_on=False)


def _oval(ax, cx, cy, w, h, text, fc=BLUE, tc="white", fs=8):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec="none", lw=0, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=3, linespacing=1.2)


def _arr(ax, x0, y0, x1, y1, label="", lbl_side="R", color=ARROW_CLR,
         lw=0.7, fs=6, fc_lbl="#666666"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=8), zorder=1)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        if abs(x1-x0) > abs(y1-y0):   # horizontal
            ax.text(mx, my+0.006, label, ha="center", va="bottom",
                    fontsize=fs, color=fc_lbl, fontstyle="italic")
        else:                          # vertical
            offx = 0.010 if lbl_side == "R" else -0.010
            ha = "left" if lbl_side == "R" else "right"
            ax.text(mx+offx, my, label, ha=ha, va="center",
                    fontsize=fs, color=fc_lbl, fontstyle="italic")


# ── flowchart ────────────────────────────────────────────────────────

def build():
    fig, ax = plt.subplots(figsize=(7.16, 10.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)

    CX     = 0.55          # decision column centre
    EGO_X  = 0.18          # ego terminal column centre
    DW     = 0.24           # diamond width
    DH     = 0.048          # diamond height
    TW     = 0.19           # terminal box width
    TH     = 0.030          # terminal box height
    VSTEP  = 0.065          # vertical step

    y = 0.97

    # ── (1) Start oval ───────────────────────────────────────────────
    _oval(ax, CX, y, 0.30, 0.032, "Takeover Clip  (N = 15,659)", fs=7.5)
    y_bot = y - 0.017

    # ── (2) Feature extraction ───────────────────────────────────────
    y -= 0.056
    fh = 0.042
    _box(ax, CX, y, 0.44, fh,
         "Feature Extraction\n42 features  ×  3 time windows\n"
         "(carState, carControl, controlsState)",
         fc=PROC_FC, fs=6.5)
    _arr(ax, CX, y_bot, CX, y + fh/2)
    y_bot = y - fh/2

    # ── (3) Cascade header ───────────────────────────────────────────
    y -= 0.048
    hh = 0.024
    _box(ax, CX, y, 0.38, hh,
         "Priority Cascade  (highest → lowest)",
         fc="#F0F0F0", ec="#AAAAAA", fs=7.5, fw="bold")
    _arr(ax, CX, y_bot, CX, y + hh/2)
    y_bot = y - hh/2

    # ── (4) Decision rows ────────────────────────────────────────────
    rows = [
        ("Stationary?\nv < 0.5 m/s\n+ driver action",
         "Ego — Stationary (G)"),
        ("Ego detector\n+ blinker active?",
         "Ego — LC/Turn + blinker"),
        ("Junction near-stop?\nv < 3 m/s + steering",
         "Ego — Junction (B1)"),
        ("Discretionary accel?\ngas + low risk + straight",
         "Ego — Accel (C)"),
        ("Ego detector,\nno Non-ego flags?",
         "Ego — Clean A/B/C"),
        ("Ego + borderline\nconflict only?",
         "Ego — Borderline (P5)"),
        ("Blinker active,\nno conflict?",
         "Ego — Blinker intent"),
    ]

    for i, (dtxt, ttxt) in enumerate(rows):
        y -= VSTEP
        # No arrow from previous
        _arr(ax, CX, y_bot, CX, y + DH/2,
             label="No" if i > 0 else "", lbl_side="R")

        # Diamond
        _diamond(ax, CX, y, DW, DH, dtxt, fs=5.8)

        # Ego terminal
        _box(ax, EGO_X, y, TW, TH, ttxt,
             fc=EGO_CLR, ec=EGO_CLR, tc="white", fs=5.8, fw="bold")

        # Yes arrow
        _arr(ax, CX - DW/2, y, EGO_X + TW/2 + 0.005, y,
             label="Yes", fc_lbl=EGO_CLR)

        y_bot = y - DH/2

    # ── (5) Default Non-ego ──────────────────────────────────────────
    y -= VSTEP * 0.82
    dh_def = TH + 0.008
    _arr(ax, CX, y_bot, CX, y + dh_def/2, label="No", lbl_side="R")
    _box(ax, CX, y, 0.22, dh_def,
         "Non-ego (default)",
         fc=NONEGO_CLR, ec=NONEGO_CLR, tc="white", fs=7.5, fw="bold")

    # ── (6) Legend ───────────────────────────────────────────────────
    y_leg = y - 0.085
    lw_b, lh_b = 0.90, 0.098
    lx_c = 0.50
    _box(ax, lx_c, y_leg, lw_b, lh_b, "", fc="#FAFAFA", ec="#CCCCCC", lw=0.5)

    ax.text(lx_c, y_leg + lh_b/2 - 0.010,
            "Detector Reference",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="#333333")

    lft = lx_c - lw_b/2 + 0.025
    yt  = y_leg + lh_b/2 - 0.026

    # Ego line
    ax.text(lft, yt, "Ego:", ha="left", va="top",
            fontsize=6, fontweight="bold", color=EGO_CLR)
    ax.text(lft + 0.032, yt,
            "G = Stationary (v<0.5)   "
            "A = Lane change (A1 blinker+LC, A2 planned, A3 conflict-trig.)   "
            "B = Junction turn (B1-B4)   "
            "C = Discretionary accel",
            ha="left", va="top", fontsize=4.8, color="#444444")
    yt -= 0.018

    # Non-ego line
    ax.text(lft, yt, "Non-ego:", ha="left", va="top",
            fontsize=6, fontweight="bold", color=NONEGO_CLR)
    ax.text(lft + 0.055, yt,
            "D = Conflict/Reactive (TTC<1.5 s, THW<0.8 s, DRAC>3, FCW)   "
            "E = Curve/ODD boundary   "
            "F = System alert",
            ha="left", va="top", fontsize=4.8, color="#444444")
    yt -= 0.018

    # Priority line
    ax.text(lft, yt, "Priority:", ha="left", va="top",
            fontsize=6, fontweight="bold", color=BLUE)
    ax.text(lft + 0.055, yt,
            "P1 Stationary > P2 Ego+blinker > P3 Junction/Accel > "
            "P4 Clean Ego > P5 Borderline > P6 Blinker only > P7 Default Non-ego",
            ha="left", va="top", fontsize=4.8, color="#444444")

    # Validation footnote
    ax.text(lx_c, y_leg - lh_b/2 + 0.010,
            "Validation: 500 clips, 4 experts, majority vote  —  "
            "Accuracy 84.0%  |  Ego P/R 90.2%/81.5%  |  "
            "Non-ego P/R 77.1%/87.5%",
            ha="center", va="center", fontsize=5.5, color="#888888",
            fontstyle="italic")

    return fig


# ── caption ──────────────────────────────────────────────────────────
CAPTION = (
    "Fig. X. Rule-based Ego / Non-ego classification flowchart (v7). "
    "Each takeover clip (N = 15,659) enters the priority cascade from top to bottom. "
    "At each level, a decision node tests whether a specific detector condition is met; "
    "if yes, the clip is labeled Ego (green) and exits the cascade. "
    "Clips not captured by any Ego rule fall through to the default Non-ego label. "
    "Ego detectors: G = stationary departure, A = lane change (3 sub-rules), "
    "B = junction/intersection turn (4 sub-rules), C = discretionary acceleration. "
    "Non-ego detectors: D = conflict/reactive (TTC, THW, DRAC, FCW), "
    "E = curve/ODD boundary, F = system alert. "
    "The classifier was validated by four human experts on a stratified random sample of 500 clips, "
    "achieving 84.0% accuracy (Ego precision/recall: 90.2%/81.5%; "
    "Non-ego precision/recall: 77.1%/87.5%)."
)


if __name__ == "__main__":
    fig = build()

    stem = OUT / "fig_ego_nonego_flowchart"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    plt.close(fig)
    print(f"Saved  {stem}.pdf / .png")

    cap_path = OUT / "fig_ego_nonego_flowchart_caption.txt"
    cap_path.write_text(CAPTION, encoding="utf-8")
    print(f"Saved  {cap_path}")
