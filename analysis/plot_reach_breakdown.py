"""Single reach-weighted graphic: topical breakdown of anti- vs pro-AI video.

Two horizontal stacked bars on a shared axis (total TikTok plays, 2026). Because
bar LENGTH is total reach, the reader sees directly that pro-AI content
out-reaches anti-AI content on net (~3:1). Each bar is segmented by theme.

Editorial styling: warm-ivory canvas, warm muted palette for the backlash bar,
cool muted palette for the positive bar, Free Systems logo, Substack link.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drilldown_common import load_tiktok, REPO

# ---- assets / config -------------------------------------------------------
LOGO = Path("/Users/andrewhall/free_systems_lab_logo transparent.png")
SUBSTACK = "freesystems.substack.com"          # <-- placeholder; confirm real URL
OUT = REPO / "analysis/reach_breakdown_2026.png"

INK = "#2c2926"
CANVAS = "#f6f1e7"
GRID = "#d8cfbf"
MUTED = "#6f6a62"

# Warm (anti) and cool (pro) muted ramps, dark->light = big->small.
WARM = ["#7c2d2a", "#a8443b", "#c0633f", "#cf8a5c", "#d9ab83",
        "#b89274", "#9a7b5f", "#8a6d52", "#6f5640"]
COOL = ["#14524e", "#1f6f68", "#2f8c80", "#4aa392", "#74bcab",
        "#4a7c8c", "#5f93a6", "#86aebd"]
LIGHT_IDX = {"warm": 4, "cool": 3}  # at/after this index use dark ink for labels

BACKLASH_LABELS = {
    "art": "art / creative theft", "deepfake": "deepfakes / misinfo",
    "jobs": "jobs / displacement", "general_neg": "general anti-AI",
    "environment": "environment / energy", "surveillance": "surveillance",
    "education": "education", "regulation": "regulation", "xrisk": "existential risk",
}
POSITIVE_LABELS = {
    "career_productivity": "career / productivity", "creative_tool": "creative tools",
    "synthetic_media_play": "synthetic-media play", "companion_affective": "AI companionship",
    "education_learning": "education / learning", "general_hype": "general hype",
    "model_stan": "model fandom", "breakthrough_science": "breakthrough science",
}


def eng(r):
    try:
        return int(float(r.get("views") or r.get("play_count") or 0))
    except (TypeError, ValueError):
        return 0


def totals(rows, key):
    d = defaultdict(int)
    for r in rows:
        d[r[key]] += eng(r)
    return dict(d)


def fmt(v):
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.0f}M"
    return f"{v/1e3:.0f}K"


def main():
    b, p = load_tiktok(only_2026=True)
    anti = sorted(totals(b, "_topic_primary").items(), key=lambda x: -x[1])
    pro = sorted(totals(p, "_theme").items(), key=lambda x: -x[1])
    anti_total = sum(v for _, v in anti)
    pro_total = sum(v for _, v in pro)
    ratio = pro_total / anti_total

    anti_colors = {t: WARM[i] for i, (t, _) in enumerate(anti)}
    pro_colors = {t: COOL[i] for i, (t, _) in enumerate(pro)}

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    fig, ax = plt.subplots(figsize=(14, 7.6))
    fig.patch.set_facecolor(CANVAS)
    ax.set_facecolor(CANVAS)

    BAR_H = 0.50
    y_pro, y_anti = 1.0, 0.0
    SCALE = 1e6  # axis in millions
    NAME_MIN = 110   # M plays: show name + % inline
    PCT_MIN = 38     # M plays: show % only

    def draw_bar(y, items, colors, labels, total, ramp):
        left = 0.0
        light_cut = LIGHT_IDX[ramp]
        for i, (t, v) in enumerate(items):
            w = v / SCALE
            ax.barh(y, w, left=left / SCALE, height=BAR_H,
                    color=colors[t], edgecolor=CANVAS, linewidth=1.4, zorder=3)
            cx = (left + v / 2) / SCALE
            txtcol = INK if i >= light_cut else "white"
            if w >= NAME_MIN:
                ax.text(cx, y, f"{labels[t]}\n{v/total*100:.0f}%", ha="center",
                        va="center", color=txtcol, fontsize=10, fontweight="bold",
                        linespacing=1.0, zorder=4)
            elif w >= PCT_MIN:
                ax.text(cx, y, f"{v/total*100:.0f}%", ha="center", va="center",
                        color=txtcol, fontsize=9.5, fontweight="bold", zorder=4)
            left += v
        ax.text(left / SCALE + 16, y, f"{fmt(total)} plays", va="center",
                ha="left", fontsize=13.5, fontweight="bold", color=INK)

    draw_bar(y_pro, pro, pro_colors, POSITIVE_LABELS, pro_total, "cool")
    draw_bar(y_anti, anti, anti_colors, BACKLASH_LABELS, anti_total, "warm")

    ax.text(-28, y_pro, "PRO-AI", va="center", ha="right", fontsize=15,
            fontweight="bold", color=COOL[0])
    ax.text(-28, y_anti, "ANTI-AI", va="center", ha="right", fontsize=15,
            fontweight="bold", color=WARM[0])

    ax.set_xlim(-300, pro_total / SCALE + 370)
    ax.set_ylim(-0.7, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels(["0", "250M", "500M", "750M", "1B"], fontsize=10, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)

    # title block
    fig.text(0.07, 0.965, "What the public actually watches about AI",
             fontsize=22, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.07, 0.908,
             "Topical breakdown of AI-related TikTok videos, 2026. Bar length = total video plays, "
             "so the reader can see\nthat pro-AI content out-reaches anti-AI content on net.",
             fontsize=11.5, color=MUTED, ha="left", va="top")

    ax.text(pro_total / SCALE * 0.5, 1.52,
            f"Pro-AI video out-reaches anti-AI video about {ratio:.1f} to 1",
            ha="center", va="center", fontsize=13.5, style="italic", color=INK)

    # legend strip (every theme, two groups)
    def legend_block(x0, items, colors, labels, total, header, hcolor):
        fig.text(x0, 0.175, header, fontsize=10.5, fontweight="bold",
                 color=hcolor, ha="left")
        for i, (t, v) in enumerate(items):
            yy = 0.145 - (i % 5) * 0.0235
            xx = x0 + (i // 5) * 0.175
            fig.patches.append(plt.Rectangle((xx, yy - 0.005), 0.013, 0.015,
                               transform=fig.transFigure, facecolor=colors[t],
                               edgecolor="none", zorder=5))
            fig.text(xx + 0.019, yy + 0.0025, f"{labels[t]}  ·  {v/total*100:.0f}%",
                     fontsize=8.4, color=INK, ha="left", va="center")

    legend_block(0.07, anti, anti_colors, BACKLASH_LABELS, anti_total,
                 "ANTI-AI THEMES", WARM[0])
    legend_block(0.56, pro, pro_colors, POSITIVE_LABELS, pro_total,
                 "PRO-AI THEMES", COOL[0])

    # logo (top-right)
    try:
        img = mpimg.imread(str(LOGO))
        ab = AnnotationBbox(OffsetImage(img, zoom=0.105), (0.95, 0.94),
                            xycoords="figure fraction", frameon=False,
                            box_alignment=(0.5, 1.0))
        fig.add_artist(ab)
        fig.text(0.95, 0.795, "FREE SYSTEMS", fontsize=9, color=MUTED,
                 ha="center", fontweight="bold")
    except Exception as e:
        print("logo skipped:", e)

    # source / substack
    fig.text(0.07, 0.03, f"Source: Free Systems Lab  ·  {SUBSTACK}",
             fontsize=9.5, color=MUTED, ha="left", style="italic")

    fig.subplots_adjust(left=0.12, right=0.965, top=0.80, bottom=0.33)
    fig.savefig(OUT, dpi=200, facecolor=CANVAS)
    print("wrote", OUT)
    print(f"anti_total={anti_total:,}  pro_total={pro_total:,}  ratio={ratio:.2f}")


if __name__ == "__main__":
    main()
