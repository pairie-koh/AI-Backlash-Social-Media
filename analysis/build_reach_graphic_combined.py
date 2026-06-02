"""Editorial reach-weighted graphic for TikTok + YouTube combined, 2026.

Sister to build_reach_graphic.py (TikTok-only). Same hand-crafted SVG +
leader-line callouts + warm/cool palette. Engagement = TikTok plays +
YouTube views, summed per theme across both platforms.

Rendered via headless Edge (Windows) since rsvg-convert isn't installed.
"""

import base64
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drilldown_common import load_tiktok, load_youtube, REPO

LOGO = Path("/Users/andrewhall/Logos/logo-dark-square.png")
SVG_OUT = REPO / "analysis/reach_breakdown_2026_combined.svg"
PNG_OUT = REPO / "analysis/reach_breakdown_2026_combined.png"

INK = "#23201c"
SUBINK = "#6f685c"
CANVAS = "#f4efe6"
HAIR = "#cabfa9"
LEADER = "#9a907c"
WARM = ["#6e2b2a", "#8f3a33", "#ab5640", "#c07a4f", "#cf9a6a",
        "#d8b487", "#c2a079", "#a98e6f", "#8f7a5f"]
COOL = ["#123f3c", "#1c5b54", "#2a786d", "#3f9384", "#62ab9b",
        "#4a7689", "#6f97a9", "#93b3c1"]

DISPLAY = "Didot, 'Hoefler Text', 'Cambria', Georgia, serif"
SANS = "'Optima', 'Avenir Next', 'Corbel', 'Helvetica Neue', sans-serif"

BACKLASH_LABELS = {
    "art": "Art / creative theft", "deepfake": "Deepfakes / misinfo",
    "jobs": "Jobs / displacement", "general_neg": "General anti-AI",
    "environment": "Environment / energy", "surveillance": "Surveillance / privacy",
    "education": "Education", "regulation": "Regulation calls", "xrisk": "Existential risk",
}
POSITIVE_LABELS = {
    "career_productivity": "Career / productivity", "creative_tool": "Creative tools",
    "synthetic_media_play": "Synthetic-media play", "companion_affective": "AI companionship",
    "education_learning": "Education / learning", "general_hype": "General hype",
    "model_stan": "Model fandom", "breakthrough_science": "Breakthrough science",
}

PRO_PLACE = {
    "synthetic_media_play": ("near", 0), "career_productivity": ("near", 0),
    "creative_tool": ("near", 0), "education_learning": ("near", 1),
    "companion_affective": ("stack", 0), "model_stan": ("stack", 1),
    "general_hype": ("stack", 2), "breakthrough_science": ("stack", 3),
}
ANTI_PLACE = {
    "art": ("near", 0), "deepfake": ("near", 0), "general_neg": ("near", 0),
    "environment": ("stack", 0), "jobs": ("stack", 1), "education": ("stack", 2),
    "regulation": ("stack", 3), "surveillance": ("stack", 4), "xrisk": ("stack", 5),
}

NEAR_X = {"art": 150, "deepfake": 300, "general_neg": 450}

W, H = 1500, 850
X0 = 120
WFULL = 900
PRO_TOP, BAR_H = 360, 70
ANTI_TOP = 566
PRO_BOT = PRO_TOP + BAR_H
ANTI_BOT = ANTI_TOP + BAR_H


def eng(r):
    try:
        return int(float(r.get("views") or r.get("play_count") or 0))
    except (TypeError, ValueError):
        return 0


def totals(rows, key):
    d = defaultdict(int)
    for r in rows:
        d[r[key]] += eng(r)
    return d


def fmt(v):
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.0f}M"
    return f"{v/1e3:.0f}K"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def render_via_edge(svg_path, png_path, w, h):
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    binary = edge if Path(edge).exists() else (chrome if Path(chrome).exists() else None)
    if not binary:
        print("no Edge/Chrome found — skipping PNG render")
        return False
    out_dir = png_path.parent
    subprocess.run([
        binary, "--headless=new", f"--screenshot={png_path}",
        f"--window-size={w},{h}", "--default-background-color=00000000",
        "--hide-scrollbars", "--force-device-scale-factor=2",
        svg_path.as_uri(),
    ], check=True, capture_output=True)
    return png_path.exists()


def main():
    tt_b, tt_p = load_tiktok(only_2026=True)
    yt_b, yt_p = load_youtube(only_2026=True)
    b = tt_b + yt_b
    p = tt_p + yt_p

    anti_d = totals(b, "_topic_primary")
    pro_d = totals(p, "_theme")
    anti = sorted(anti_d.items(), key=lambda x: -x[1])
    pro = sorted(pro_d.items(), key=lambda x: -x[1])
    anti_total = sum(v for _, v in anti)
    pro_total = sum(v for _, v in pro)
    ratio = pro_total / anti_total
    scale = WFULL / pro_total
    anti_colors = {t: WARM[i] for i, (t, _) in enumerate(anti)}
    pro_colors = {t: COOL[i] for i, (t, _) in enumerate(pro)}

    S = []
    S.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{CANVAS}"/>')

    for val, lab in [(1e9, "1B"), (2e9, "2B"), (3e9, "3B"), (4e9, "4B")]:
        if val > pro_total:
            break
        gx = X0 + val * scale
        S.append(f'<line x1="{gx:.1f}" y1="{PRO_TOP-14}" x2="{gx:.1f}" y2="{ANTI_BOT+14}" '
                 f'stroke="{HAIR}" stroke-width="1"/>')
        S.append(f'<text x="{gx:.1f}" y="{PRO_TOP-22}" font-family="{SANS}" font-size="12.5" '
                 f'fill="{SUBINK}" text-anchor="middle">{lab}</text>')

    def draw(items, colors, labels, place, total, bar_top, side):
        bot = bar_top + BAR_H
        centers = {}
        left = X0
        for t, v in items:
            w = v * scale
            centers[t] = (left + w / 2, w, v)
            r = 3 if w > 6 else 1
            S.append(f'<rect x="{left:.1f}" y="{bar_top}" width="{max(w,1.2):.1f}" height="{BAR_H}" '
                     f'fill="{colors[t]}" rx="{min(r,3)}"/>')
            left += w
            S.append(f'<line x1="{left:.1f}" y1="{bar_top}" x2="{left:.1f}" y2="{bot}" '
                     f'stroke="{CANVAS}" stroke-width="1.6"/>')
        bar_end = left
        S.append(f'<text x="{bar_end+18:.1f}" y="{bar_top+BAR_H/2+6:.1f}" font-family="{SANS}" '
                 f'font-size="20" font-weight="700" fill="{INK}">{fmt(total)} views+plays</text>')

        near_y = {0: (bar_top - 70) if side == "pro" else (bot + 64),
                  1: (bar_top - 122) if side == "pro" else (bot + 116)}
        stack_x = (1175 if side == "pro" else bar_end + 150)
        stack_y0 = (212 if side == "pro" else bar_top - 56)
        stack_dy = 43 if side == "pro" else 39

        for t, v in items:
            cx, w, val = centers[t]
            if t not in place:
                continue
            kind, idx = place[t]
            pct = round(val / total * 100)
            color = colors[t]
            if kind == "near":
                ly = near_y[idx]
                edge = bar_top if side == "pro" else bot
                lx = NEAR_X.get(t, cx)
                tipy = ly + (26 if side == "pro" else -26)
                S.append(f'<polyline points="{cx:.1f},{edge} {lx:.1f},{tipy:.1f}" '
                         f'fill="none" stroke="{LEADER}" stroke-width="1.1"/>')
                S.append(f'<circle cx="{cx:.1f}" cy="{edge}" r="3" fill="{color}"/>')
                pct_y = ly + 24
                S.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="{SANS}" font-size="14.5" '
                         f'fill="{INK}" text-anchor="middle" letter-spacing="0.4">{esc(labels[t])}</text>')
                S.append(f'<text x="{lx:.1f}" y="{pct_y:.1f}" font-family="{DISPLAY}" font-size="22" '
                         f'font-weight="700" fill="{color}" text-anchor="middle">{pct}%</text>')
            else:
                ry = stack_y0 + idx * stack_dy
                edge = bar_top if side == "pro" else (bar_top + BAR_H / 2)
                tx = stack_x
                elbow = edge - 14 if side == "pro" else edge
                S.append(f'<polyline points="{cx:.1f},{edge:.1f} {cx:.1f},{elbow:.1f} '
                         f'{tx-12:.1f},{ry:.1f}" fill="none" stroke="{LEADER}" stroke-width="1.1"/>')
                S.append(f'<circle cx="{cx:.1f}" cy="{edge:.1f}" r="2.6" fill="{color}"/>')
                S.append(f'<rect x="{tx:.1f}" y="{ry-9:.1f}" width="12" height="12" fill="{color}" rx="2"/>')
                S.append(f'<text x="{tx+19:.1f}" y="{ry+1:.1f}" font-family="{SANS}" font-size="14" '
                         f'fill="{INK}" letter-spacing="0.3">{esc(labels[t])}'
                         f'<tspan font-family="{DISPLAY}" font-weight="700" fill="{INK}" '
                         f'dx="6">{pct}%</tspan></text>')

    draw(pro, pro_colors, POSITIVE_LABELS, PRO_PLACE, pro_total, PRO_TOP, "pro")
    draw(anti, anti_colors, BACKLASH_LABELS, ANTI_PLACE, anti_total, ANTI_TOP, "anti")

    S.append(f'<text x="{X0-22}" y="{PRO_TOP+BAR_H/2+7:.1f}" font-family="{DISPLAY}" font-size="22" '
             f'font-weight="700" fill="{COOL[0]}" text-anchor="end">PRO-AI</text>')
    S.append(f'<text x="{X0-22}" y="{ANTI_TOP+BAR_H/2+7:.1f}" font-family="{DISPLAY}" font-size="22" '
             f'font-weight="700" fill="{WARM[0]}" text-anchor="end">ANTI-AI</text>')

    S.append(f'<text x="{X0-2}" y="84" font-family="{DISPLAY}" font-size="46" font-weight="700" '
             f'fill="{INK}">What the public actually watches about AI</text>')
    S.append(f'<text x="{X0-2}" y="112" font-family="{SANS}" font-size="16" '
             f'fill="{SUBINK}" letter-spacing="0.4">TikTok &#43; YouTube, 2026 &#160;&#160;&#183;&#160;&#160;'
             f'reach-weighted by total views &#43; plays</text>')
    S.append(f'<line x1="{X0-2}" y1="128" x2="{W-90}" y2="128" stroke="{HAIR}" stroke-width="1"/>')

    L = 78
    lx = W - 52 - L
    ly = H - 18 - L
    cy = ly + L / 2
    rule_y = ly - 8
    S.append(f'<line x1="{X0-2}" y1="{rule_y:.1f}" x2="{lx-34:.1f}" y2="{rule_y:.1f}" '
             f'stroke="{HAIR}" stroke-width="1"/>')
    S.append(f'<text x="{X0-2}" y="{cy-16:.1f}" font-family="{DISPLAY}" font-size="17" '
             f'font-weight="700" fill="{INK}">Free Systems Lab</text>')
    S.append(f'<text x="{X0-2}" y="{cy+5:.1f}" font-family="{SANS}" font-size="13.5" '
             f'fill="{SUBINK}" letter-spacing="0.3">Stanford Graduate School of Business'
             f'&#160;&#160;&#183;&#160;&#160;Hoover Institution</text>')
    S.append(f'<text x="{X0-2}" y="{cy+26:.1f}" font-family="{SANS}" font-size="13.5" '
             f'fill="{INK}">freesystems.substack.com</text>')
    if LOGO.exists():
        data = base64.b64encode(LOGO.read_bytes()).decode()
        S.append(f'<image x="{lx:.1f}" y="{ly:.1f}" width="{L}" height="{L}" '
                 f'xlink:href="data:image/png;base64,{data}"/>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n' + "\n".join(S) + "\n</svg>\n")
    SVG_OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {SVG_OUT}")
    print(f"anti={anti_total:,} pro={pro_total:,} ratio={ratio:.2f}x")

    if render_via_edge(SVG_OUT, PNG_OUT, W, H):
        print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
