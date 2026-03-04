#!/usr/bin/env python3
"""
make_fig_taxonomy.py
=====================
Publication-quality 3-panel "Taxonomy Summary" figure for the TakeOver paper.
  Row 1: Panel (a) Alluvial/Sankey — full width
  Row 2: Panel (b) Trigger composition by source  |  Panel (c) Trigger co-activation (UpSet)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
import pandas as pd
from pathlib import Path as PPath

# ── Paths ──
ROOT  = PPath("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE  = ROOT / "Code"
STATS = CODE / "stats_output"
OUT   = CODE / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Color palette ──
C = dict(
    blue="#4C78A8", orange="#F58518", red="#E45756", teal="#72B7B2",
    green="#54A24B", purple="#B279A2", gray="#BAB0AC", brown="#9D755D",
)

TRIG_COLORS = {
    "Steering Override": C["blue"],  "Brake Override": C["teal"],
    "Gas Override": C["green"],      "System / Unknown": C["gray"],
}
TRIG_ORDER = ["Steering Override", "Brake Override", "Gas Override", "System / Unknown"]
TRIG_SHORT = {
    "Steering Override": "Steering", "Brake Override": "Brake",
    "Gas Override": "Gas", "System / Unknown": "System",
}

SRC_COLORS = {
    "OEM-only": "#E8943A", "OP-only": C["purple"],
    "Both-active": "#7A6C5D", "Other": "#D5D0CB",
}
SRC_ORDER = ["OEM-only", "OP-only", "Both-active", "Other"]

# Channel colors
CH_COLORS = {"S": C["blue"], "B": C["teal"], "G": C["green"]}
CH_LABELS = {"S": "Steering", "B": "Brake", "G": "Gas"}

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": False,
})

# ═══════════════════════════════════════════════════════════════
# Load & merge
# ═══════════════════════════════════════════════════════════════
pc = pd.read_csv(STATS / "per_clip.csv")
es = pd.read_csv(STATS / "engagement_source.csv")
df = pc.merge(
    es[["car_model", "dongle_id", "route_id", "clip_id", "source"]],
    on=["car_model", "dongle_id", "route_id", "clip_id"], how="left",
)
df = df[df["is_noise"] == False].copy()

def simplify_src(s):
    if s in ("oem_only", "oem_primary"):  return "OEM-only"
    elif s in ("openpilot_only", "openpilot_primary"): return "OP-only"
    elif s == "both": return "Both-active"
    else: return "Other"

df["src"] = df["source"].apply(simplify_src)
df["override"] = df["n_triggers"] > 0
N = len(df)
print(f"Total clean clips: {N:,}")

# ═══════════════════════════════════════════════════════════════
# Sankey helper
# ═══════════════════════════════════════════════════════════════
def sankey_band(ax, x0, y0t, y0b, x1, y1t, y1b, color, alpha=0.38):
    xm = (x0 + x1) / 2
    verts = [
        (x0, y0t), (xm, y0t), (xm, y1t), (x1, y1t),
        (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0t),
    ]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(
        Path(verts, codes), facecolor=color, edgecolor="none",
        alpha=alpha, linewidth=0, zorder=2))


# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.16, 7.2))

ax_a     = fig.add_axes([0.10, 0.56, 0.85, 0.37])   # Sankey
ax_b     = fig.add_axes([0.08, 0.08, 0.26, 0.37])    # Stacked bars
ax_c_bar = fig.add_axes([0.48, 0.24, 0.48, 0.21])    # UpSet bars
ax_c_dot = fig.add_axes([0.48, 0.08, 0.48, 0.15])    # UpSet dots


# ═══════════════════════════════════════════════════════════════
# PANEL (a): Sankey
# ═══════════════════════════════════════════════════════════════
ax_a.set_xlim(-0.01, 1.01)
ax_a.set_ylim(-0.03, 1.10)
ax_a.axis("off")
ax_a.text(-0.01, 1.12, "(a) Takeover taxonomy flow", fontsize=10,
          fontweight="bold", va="bottom")

LX = [0.03, 0.43, 0.77]
LW = [0.06, 0.06, 0.06]
GAP = 0.022; MIN_H = 0.022

src_counts = {s: (df["src"] == s).sum() for s in SRC_ORDER}
ovr_labels = ["Override", "Non-override"]
ovr_counts = {"Override": int(df["override"].sum()),
              "Non-override": int((~df["override"]).sum())}
trig_counts = {t: (df["primary_trigger"] == t).sum() for t in TRIG_ORDER}

src_ovr = {}
for s in SRC_ORDER:
    for o in ovr_labels:
        mo = df["override"] if o == "Override" else ~df["override"]
        src_ovr[(s, o)] = int(((df["src"] == s) & mo).sum())

ovr_trig = {}
for o in ovr_labels:
    for t in TRIG_ORDER:
        mo = df["override"] if o == "Override" else ~df["override"]
        ovr_trig[(o, t)] = int((mo & (df["primary_trigger"] == t)).sum())

def layout_nodes(counts, order, total, gap, min_h):
    n = len(order); usable = 1.0 - gap * (n - 1)
    raw = {k: max(counts[k] / total * usable, min_h) for k in order}
    scale = usable / sum(raw.values())
    heights = {k: raw[k] * scale for k in order}
    pos = {}; y = 1.0
    for k in order:
        pos[k] = (y, y - heights[k]); y -= heights[k] + gap
    return pos

src_pos  = layout_nodes(src_counts,  SRC_ORDER,  N, GAP, MIN_H)
ovr_pos  = layout_nodes(ovr_counts,  ovr_labels, N, GAP, MIN_H)
trig_pos = layout_nodes(trig_counts, TRIG_ORDER, N, GAP, MIN_H)

def draw_rect(ax, x, w, yt, yb, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, yb), w, yt - yb, boxstyle="round,pad=0.005",
        facecolor=color, edgecolor="white", linewidth=0.8, zorder=4))

def fmt(name, count, total):
    return f"{name}   {count:,} ({count/total*100:.1f}%)"

for s in SRC_ORDER:
    yt, yb = src_pos[s]
    draw_rect(ax_a, LX[0], LW[0], yt, yb, SRC_COLORS[s])
    ax_a.text(LX[0]-0.015, (yt+yb)/2, fmt(s, src_counts[s], N),
              fontsize=7.5, va="center", ha="right", color="#333")

ovr_colors = {"Override": "#555555", "Non-override": "#CCCCCC"}
for o in ovr_labels:
    yt, yb = ovr_pos[o]
    draw_rect(ax_a, LX[1], LW[1], yt, yb, ovr_colors[o])
    ax_a.text(LX[1]+LW[1]+0.015, (yt+yb)/2, fmt(o, ovr_counts[o], N),
              fontsize=7, va="center", ha="left", color="#444")

for t in TRIG_ORDER:
    yt, yb = trig_pos[t]
    draw_rect(ax_a, LX[2], LW[2], yt, yb, TRIG_COLORS[t])
    ax_a.text(LX[2]+LW[2]+0.015, (yt+yb)/2,
              fmt(TRIG_SHORT[t], trig_counts[t], N),
              fontsize=7.5, va="center", ha="left", color="#333")

for lbl, x, w in [("Engagement Source", LX[0], LW[0]),
                   ("Override Detected", LX[1], LW[1]),
                   ("Primary Trigger",   LX[2], LW[2])]:
    ax_a.text(x+w/2, 1.055, lbl, fontsize=7.5, ha="center", va="bottom",
              fontweight="bold", color="#444")

# Flows
ovr_doff = {o: 0.0 for o in ovr_labels}
for s in SRC_ORDER:
    syt, syb = src_pos[s]; sh = syt-syb; so = 0.0
    for o in ovr_labels:
        cnt = src_ovr[(s, o)]
        if not cnt: continue
        hs = sh*cnt/src_counts[s]
        oyt = ovr_pos[o][0]; oh = oyt-ovr_pos[o][1]
        ho = oh*cnt/ovr_counts[o]
        sankey_band(ax_a, LX[0]+LW[0], syt-so, syt-so-hs,
                    LX[1], oyt-ovr_doff[o], oyt-ovr_doff[o]-ho,
                    SRC_COLORS[s], 0.28)
        so += hs; ovr_doff[o] += ho

trig_doff = {t: 0.0 for t in TRIG_ORDER}
for o in ovr_labels:
    oyt = ovr_pos[o][0]; oh = oyt-ovr_pos[o][1]; oo = 0.0
    for t in TRIG_ORDER:
        cnt = ovr_trig[(o, t)]
        if not cnt: continue
        ho = oh*cnt/ovr_counts[o]
        tyt = trig_pos[t][0]; th = tyt-trig_pos[t][1]
        ht = th*cnt/trig_counts[t]
        sankey_band(ax_a, LX[1]+LW[1], oyt-oo, oyt-oo-ho,
                    LX[2], tyt-trig_doff[t], tyt-trig_doff[t]-ht,
                    TRIG_COLORS[t], 0.32)
        oo += ho; trig_doff[t] += ht


# ═══════════════════════════════════════════════════════════════
# PANEL (b): 100% stacked bars
# ═══════════════════════════════════════════════════════════════
ax_b.set_title("(b) Trigger composition\n     by source", fontsize=10,
               fontweight="bold", loc="left", pad=4)

bar_srcs = ["OEM-only", "OP-only", "Both-active"]
bar_x = np.arange(3); bar_w = 0.62
bottoms = np.zeros(3)

for t in TRIG_ORDER:
    heights = np.array([
        ((df["src"]==s) & (df["primary_trigger"]==t)).sum()
        / (df["src"]==s).sum() * 100 for s in bar_srcs])
    ax_b.bar(bar_x, heights, bar_w, bottom=bottoms,
             color=TRIG_COLORS[t], edgecolor="white", linewidth=0.5,
             label=TRIG_SHORT[t], zorder=3)
    for i, (h, b) in enumerate(zip(heights, bottoms)):
        if h >= 7:
            ax_b.text(bar_x[i], b+h/2, f"{h:.0f}%", ha="center", va="center",
                      fontsize=7, color="white", fontweight="bold", zorder=4)
    bottoms += heights

for i, s in enumerate(bar_srcs):
    n = (df["src"]==s).sum()
    ax_b.text(bar_x[i], 103, f"n={n:,}", ha="center", va="bottom",
              fontsize=6.5, color="#555", fontstyle="italic")

ax_b.set_ylim(0, 115); ax_b.set_xlim(-0.5, 2.5)
ax_b.set_xticks(bar_x)
ax_b.set_xticklabels(["OEM-only", "OP-only", "Both-active"], fontsize=7.5)
ax_b.set_ylabel("Percentage (%)", fontsize=9)
ax_b.set_yticks([0, 25, 50, 75, 100])
ax_b.legend(loc="lower center", frameon=True, fancybox=True,
            framealpha=0.92, edgecolor="#ddd", fontsize=6.5,
            ncol=2, handlelength=1.0, handletextpad=0.3,
            columnspacing=0.6, bbox_to_anchor=(0.5, -0.20))
for sp in ["top", "right"]:
    ax_b.spines[sp].set_visible(False)


# ═══════════════════════════════════════════════════════════════
# PANEL (c): Trigger co-activation (UpSet-style)
# ═══════════════════════════════════════════════════════════════
def get_combo(row):
    p = []
    if row["trig_steer"]: p.append("S")
    if row["trig_brake"]: p.append("B")
    if row["trig_gas"]:   p.append("G")
    return tuple(p) if p else ("None",)

df["combo"] = df.apply(get_combo, axis=1)
combo_counts = df["combo"].value_counts()

# Sort by group (none → single → dual → triple), then by count within group
def combo_sort_key(combo):
    n = 0 if combo == ("None",) else len(combo)
    return (n, -combo_counts[combo])

combos_sorted = sorted(combo_counts.index.tolist(), key=combo_sort_key)
n_combos = len(combos_sorted)
counts_sorted = [combo_counts[c] for c in combos_sorted]

channels = ["S", "B", "G"]
ch_y = {"S": 2, "B": 1, "G": 0}

# Bar colors: single → channel color; dual → orange; triple → red; none → gray
def bar_color(combo):
    if combo == ("None",): return C["gray"]
    if len(combo) == 1: return CH_COLORS[combo[0]]
    if len(combo) == 2: return C["orange"]
    return C["red"]

# ── Bar chart ──
ax_c_bar.set_title("(c) Trigger co-activation patterns", fontsize=10,
                   fontweight="bold", loc="left", pad=8)

x_pos = np.arange(n_combos)
colors_list = [bar_color(c) for c in combos_sorted]

ax_c_bar.bar(x_pos, counts_sorted, width=0.62,
             color=colors_list, edgecolor="white", linewidth=0.5, zorder=3)

for i, (cnt, combo) in enumerate(zip(counts_sorted, combos_sorted)):
    pct = cnt / N * 100
    ax_c_bar.text(x_pos[i], cnt + 50, f"{cnt:,}\n({pct:.1f}%)",
                  ha="center", va="bottom", fontsize=6, color="#333",
                  linespacing=1.2)

ax_c_bar.set_xlim(-0.6, n_combos - 0.4)
ax_c_bar.set_ylim(0, max(counts_sorted) * 1.32)
ax_c_bar.set_ylabel("Count", fontsize=8)
ax_c_bar.set_xticks([])
ax_c_bar.spines["bottom"].set_visible(False)
ax_c_bar.tick_params(bottom=False)
for sp in ["top", "right"]:
    ax_c_bar.spines[sp].set_visible(False)

# ── Dot matrix ──
ax_c_dot.set_xlim(-0.6, n_combos - 0.4)
ax_c_dot.set_ylim(-0.8, 2.8)
ax_c_dot.axis("off")

# Channel labels
for ch in channels:
    ax_c_dot.text(-0.55, ch_y[ch], CH_LABELS[ch], fontsize=7.5,
                  ha="right", va="center", color=CH_COLORS[ch], fontweight="bold")

# Horizontal grid
for ch in channels:
    ax_c_dot.axhline(ch_y[ch], color="#F0F0F0", lw=0.6, zorder=1)

# Dots and connectors
for i, combo in enumerate(combos_sorted):
    active = set(combo) if combo != ("None",) else set()
    for ch in channels:
        if ch in active:
            ax_c_dot.scatter(x_pos[i], ch_y[ch], s=80,
                            color=CH_COLORS[ch], edgecolors="white",
                            linewidth=0.5, zorder=5)
        else:
            ax_c_dot.scatter(x_pos[i], ch_y[ch], s=25,
                            color="#E0E0E0", edgecolors="none", zorder=4)
    if len(active) >= 2:
        ys = sorted([ch_y[c] for c in active])
        ax_c_dot.plot([x_pos[i]]*2, [ys[0], ys[-1]],
                     color="#555", lw=1.8, zorder=3, solid_capstyle="round")

# Group brackets below
n_groups = {}
for i, combo in enumerate(combos_sorted):
    n = 0 if combo == ("None",) else len(combo)
    n_groups.setdefault(n, []).append(i)

grp_labels = {0: "No override", 1: "Single channel",
              2: "Dual channels", 3: "All three"}
for n, indices in sorted(n_groups.items()):
    xmid = np.mean(indices)
    total_g = sum(counts_sorted[i] for i in indices)
    pct_g = total_g / N * 100
    by = -0.6
    ax_c_dot.text(xmid, by, f"{grp_labels[n]}\n{total_g:,} ({pct_g:.1f}%)",
                 fontsize=5.8, ha="center", va="top", color="#666",
                 linespacing=1.3)
    if len(indices) > 1:
        x_lo, x_hi = min(indices)-0.3, max(indices)+0.3
        bk = -0.35
        ax_c_dot.plot([x_lo, x_lo, x_hi, x_hi],
                     [bk+0.12, bk, bk, bk+0.12],
                     color="#BBB", lw=0.6, clip_on=False)


# ═══════════════════════════════════════════════════════════════
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_taxonomy.{ext}")
plt.close()
print(f"\nSaved fig_taxonomy.pdf/png → {OUT}")
