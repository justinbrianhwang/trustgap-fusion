"""Fig. 1 (architecture) as an editable SVG: large type, keywords only; explanations belong in the caption."""
import base64, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
W, H = 1600, 800
BLUE, BLUE_F = "#2F6FB3", "#DCEBFA"
ORG, ORG_F = "#D9772B", "#FDE8D2"
GRN, GRN_F = "#2E8B4E", "#DDF2E1"
GRY, GRY_F = "#666666", "#EEEEEE"
PUR, PUR_F = "#6B4FA3", "#EBE4F6"
RED, INK = "#C0392B", "#333333"
FONT = "font-family='Helvetica, Arial, sans-serif'"
el = []

def img64(p): return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
def rect(x, y, w, h, fill, stroke, sw=2, r=8, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    el.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{r}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{d}/>")
def text(x, y, s, size=14, weight="normal", anchor="middle", fill=INK, style="normal"):
    el.append(f"<text x='{x}' y='{y}' font-size='{size}' font-weight='{weight}' font-style='{style}' text-anchor='{anchor}' fill='{fill}' {FONT}>{s}</text>")
def box(x, y, w, h, fill, stroke, title, sub=None, ts=17, ss=13, sw=2, snow=False, tcol=INK):
    rect(x, y, w, h, fill, stroke, sw)
    cy = y + h / 2 + (ts / 3 if not sub else -2)
    text(x + w / 2, cy, title, ts, "bold", fill=tcol)
    if sub: text(x + w / 2, cy + ss + 5, sub, ss, fill="#444444")
    if snow: text(x + 14, y + 20, "&#10052;", 16, fill="#3B7DD8")
def arrow(x1, y1, x2, y2, color=INK, sw=2.2, dash=None, m="a", label=None, lx=None, ly=None, size=13, anchor="middle"):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    el.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{sw}'{d} marker-end='url(#{m})'/>")
    if label: text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6, label, size, fill=color, anchor=anchor)
def path(d, color=INK, sw=2.2, dash=None, m="a"):
    dd = f" stroke-dasharray='{dash}'" if dash else ""
    el.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{sw}'{dd} marker-end='url(#{m})'/>")
def image(x, y, w, h, p):
    el.append(f"<image x='{x}' y='{y}' width='{w}' height='{h}' href='{img64(p)}' preserveAspectRatio='none'/>")
    el.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='none' stroke='{INK}' stroke-width='1'/>")
def header(x, y, s, col="#222222"): text(x, y, s, 18, "bold", "start", col)
def bolt(x, y, s): text(x, y, "&#9889;", 15, fill="#C9A227"); text(x + 12, y, s, 13, anchor="start", fill="#7A5E00")

# ---------------- (A) sensors
header(30, 40, "Sensors")
rect(30, 55, 235, 250, BLUE_F, BLUE)
text(147, 77, "Cameras", 16, "bold", fill=BLUE)
image(42, 88, 136, 136, "fig1_front_224.png"); text(110, 243, "Front 224²", 13)
for i, (name, f) in enumerate([("Left 128²", "fig1_left_128.png"), ("Right 128²", "fig1_right_128.png"), ("Center 128²", "fig1_center_crop_128.png")]):
    y = 88 + i * 70
    image(192, y, 56, 56, f); text(220, y + 69, name, 11.5)
bolt(36, 328, "noise · blur · dark · occlusion")
rect(30, 345, 235, 165, ORG_F, ORG)
text(147, 367, "LiDAR", 16, "bold", fill=ORG)
image(42, 377, 211, 103, "fig1_lidar_bev.png"); text(147, 498, "BEV histogram 224²", 13)
bolt(36, 533, "dropout · jitter · sector loss")

# ---------------- (B) tokenizer
header(300, 40, "Tokenizer (frozen)")
box(300, 130, 175, 64, GRY_F, GRY, "ResNet-50", "4 camera views", snow=True)
box(300, 385, 175, 64, GRY_F, GRY, "ResNet-18", "LiDAR", snow=True)
box(300, 250, 175, 72, GRY_F, GRY, "Patch + view +", "position embedding", ts=15, ss=15)
arrow(265, 162, 300, 162, BLUE, m="ab"); arrow(265, 417, 300, 417, ORG, m="ao")
arrow(387, 194, 387, 250, BLUE, m="ab"); arrow(387, 385, 387, 322, ORG, m="ao")

# ---------------- (C) tokens
header(515, 40, "Tokens")
x0, y0, wt, hh = 520, 268, 220, 32
groups = [(50, BLUE, BLUE_F), (17, BLUE, BLUE_F), (17, BLUE, BLUE_F), (17, BLUE, BLUE_F), (50, ORG, ORG_F)]
x = x0
for n, s, f in groups:
    w = n * wt / 151
    rect(x, y0, w - 1.5, hh, f, s, 1.2, r=3)
    rect(x + w - 6, y0 + 3, 4, hh - 6, s, s, 0.5, r=1)
    x += w
text(x0 + 101 * wt / 151 / 2, y0 + 50, "camera 101", 13, fill=BLUE)
text(x0 + (101 + 25) * wt / 151, y0 + 50, "LiDAR 50", 13, fill=ORG)
text(x0 + wt / 2, y0 + 74, "151 tokens, d = 256", 15, "bold")
arrow(475, 286, 520, 284); arrow(740, 284, 790, 284)
arrow(630, 302, 630, 470, GRN, 2.8, m="ag", label="features", lx=660, ly=400, anchor="start")

# ---------------- (D) fusion
header(790, 40, "Gated fusion")
box(790, 240, 180, 90, GRY_F, GRY, "Encoder ×6", "self-attention")
box(1010, 240, 180, 90, GRY_F, GRY, "Decoder ×6", "cross-attention")
box(1010, 120, 180, 60, "#FFFFFF", GRY, "411 queries", "400 grid + 1 + 10", ts=15, ss=12, sw=1.5)
arrow(1100, 180, 1100, 240, sw=1.8)
arrow(970, 285, 1010, 285)
for cx in (880, 1100):
    el.append(f"<polygon points='{cx},321 {cx+10},331 {cx},341 {cx-10},331' fill='{GRN}'/>")

# ---------------- (E) outputs
header(1240, 40, "Outputs")
rect(1240, 100, 320, 345, PUR_F, PUR)
text(1400, 126, "Traffic grid 20×20×7", 17, "bold", fill=PUR)
gx, gy, cs = 1265, 140, 13.5
hot = {(3, 9), (3, 10), (4, 9), (4, 10), (9, 4), (9, 5), (10, 4), (13, 14), (14, 14), (14, 15), (17, 9), (17, 10)}
warm = {(r + dr, c + dc) for r, c in hot for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
for r in range(20):
    for c in range(20):
        col = PUR if (r, c) in hot else ("#B8A6DA" if (r, c) in warm else "#FFFFFF")
        el.append(f"<rect x='{gx + c * cs}' y='{gy + r * cs}' width='{cs}' height='{cs}' fill='{col}' stroke='#D8CFEA' stroke-width='0.6'/>")
el.append(f"<polygon points='{gx + 135},{gy + 270 - 3} {gx + 126},{gy + 270 + 10} {gx + 144},{gy + 270 + 10}' fill='{INK}'/>")
text(1400, 435, "score · offset · size · yaw · speed", 12.5, fill="#444444")
arrow(1190, 285, 1240, 285)
rect(1240, 458, 320, 34, "#F6F3FA", "#9C8CC2", 1.5)
text(1400, 480, "waypoints · junction · light · stop", 12.5, fill="#555555")
path("M 1190 300 L 1215 300 L 1215 475 L 1240 475", "#9C8CC2", 1.5, m="ap")

# ---------------- proposed branch
header(300, 500, "Reliability estimation and gating", GRN)
box(300, 515, 190, 70, GRN_F, GRN, "Statistics", "token stats + raw cues", sw=2.5, tcol=GRN)
box(540, 515, 190, 70, GRN_F, GRN, "Estimator", "MLP", sw=2.5, tcol=GRN)
arrow(490, 550, 540, 550, GRN, 2.8, m="ag")
path("M 147 512 L 147 565 L 300 565", GRN, 2, m="ag"); text(222, 559, "raw cues", 12, fill=GRN)
for cx, lab in ((795, "q_cam"), (865, "q_lidar")):
    el.append(f"<circle cx='{cx}' cy='550' r='28' fill='{GRN_F}' stroke='{GRN}' stroke-width='2.5'/>")
    text(cx, 555, lab, 14, "bold", fill=GRN)
arrow(730, 550, 767, 550, GRN, 2.8, m="ag")
rect(930, 500, 330, 120, GRN_F, GRN, 3)
text(1095, 526, "Reliability gate", 18, "bold", fill=GRN)
text(950, 556, "(a) log-prior", 15, "bold", "start"); text(1240, 556, "+ α·log q", 15, anchor="end", style="italic")
text(950, 581, "(b) token scaling", 15, "bold", "start"); text(1240, 581, "q · x", 15, anchor="end", style="italic")
text(950, 606, "(c) key scaling", 15, "bold", "start"); text(1240, 606, "q^γ · k", 15, anchor="end", style="italic")
arrow(893, 550, 930, 550, GRN, 2.8, m="ag")
path("M 980 500 L 980 420 L 880 420 L 880 345", GRN, 3, m="ag")
path("M 1100 500 L 1100 345", GRN, 3, m="ag")
text(1030, 412, "keys", 14, fill=GRN); text(1120, 430, "keys", 14, "normal", "start", GRN)

# ---------------- training-only band
rect(30, 640, 1540, 140, "none", RED, 1.8, r=10, dash="10,7")
header(44, 664, "Training only", RED)
box(60, 685, 300, 72, GRY_F, GRY, "Frozen reference", "one modality masked", snow=True)
box(410, 685, 200, 72, "#FFFFFF", GRY, "Error  E", "vs. ground truth", sw=1.5)
box(660, 685, 260, 72, "#FFFFFF", GRY, "q* = exp(−E / τ)", None, ts=18, sw=1.5)
box(1000, 685, 200, 72, "#FFFFFF", GRY, "L_task", "InterFuser loss", sw=1.5)
box(1250, 685, 300, 72, "#FFFFFF", INK, "L = L_task + λ·L_quality", None, ts=18, sw=2.5)
arrow(360, 721, 410, 721, RED, 2, "8,6", "ar"); arrow(610, 721, 660, 721, RED, 2, "8,6", "ar"); arrow(1200, 721, 1250, 721, RED, 2, "8,6", "ar")
path("M 200 575 L 200 685", RED, 2, "8,6", "ar"); text(210, 625, "same input", 13, anchor="start", fill=RED)
path("M 830 685 L 830 580", RED, 2.2, "8,6", "ar"); text(842, 668, "L_quality", 15, "bold", "start", RED)
path("M 1400 445 L 1400 455 L 1580 455 L 1580 631 L 1100 631 L 1100 685", RED, 2, "8,6", "ar")
path("M 920 740 L 1000 740", RED, 2, "8,6", "ar")

# ---------------- legend
lx, ly = 1300, 500
rect(lx, ly, 260, 120, "#FFFFFF", "#BBBBBB", 1.2)
for i, (f, s, n) in enumerate([(BLUE_F, BLUE, "camera"), (ORG_F, ORG, "LiDAR"), (GRN_F, GRN, "proposed"), (GRY_F, GRY, "upstream"), (PUR_F, PUR, "output")]):
    rect(lx + 12, ly + 12 + i * 21, 20, 14, f, s, 1.2, r=3); text(lx + 40, ly + 24 + i * 21, n, 13, anchor="start")
text(lx + 165, ly + 24, "&#10052; frozen", 13, anchor="start")
text(lx + 165, ly + 45, "&#9889; degradation", 13, anchor="start")
el.append(f"<line x1='{lx+165}' y1='{ly+62}' x2='{lx+190}' y2='{ly+62}' stroke='{RED}' stroke-width='2' stroke-dasharray='8,6'/>"); text(lx + 196, ly + 66, "training", 13, anchor="start")
el.append(f"<line x1='{lx+165}' y1='{ly+83}' x2='{lx+190}' y2='{ly+83}' stroke='{GRN}' stroke-width='3'/>"); text(lx + 196, ly + 87, "gate signal", 13, anchor="start")
el.append(f"<polygon points='{lx+177},{ly+97} {lx+185},{ly+105} {lx+177},{ly+113} {lx+169},{ly+105}' fill='{GRN}'/>"); text(lx + 196, ly + 109, "gate entry", 13, anchor="start")

def marker(i, c): return f"<marker id='{i}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{c}'/></marker>"
defs = "<defs>" + "".join(marker(i, c) for i, c in (("a", INK), ("ab", BLUE), ("ao", ORG), ("ag", GRN), ("ar", RED), ("ap", "#9C8CC2"))) + "</defs>"
svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>\n<rect width='{W}' height='{H}' fill='white'/>\n{defs}\n" + "\n".join(el) + "\n</svg>\n"
open("fig1_architecture.svg", "w", encoding="utf-8").write(svg)
print("svg written", len(svg) // 1024, "KB")
