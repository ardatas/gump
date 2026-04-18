"""gump_pitch.pptx — English, unified design, rainforest-first cover."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).parent
IMG = HERE / "images"
OUT = HERE / "gump_pitch.pptx"

W, H = Inches(13.333), Inches(7.5)

# ═══ DESIGN TOKENS ═══
BG          = RGBColor(0x08, 0x0C, 0x0A)
BG_DEEP     = RGBColor(0x04, 0x06, 0x05)
PANEL       = RGBColor(0x10, 0x17, 0x14)
PANEL_HI    = RGBColor(0x18, 0x22, 0x1D)
BORDER      = RGBColor(0x1F, 0x2A, 0x24)
BORDER_HI   = RGBColor(0x33, 0x44, 0x3B)
TEXT        = RGBColor(0xEC, 0xEF, 0xED)
MUTED       = RGBColor(0x8A, 0x96, 0x91)
DIM         = RGBColor(0x50, 0x5C, 0x56)
ACCENT      = RGBColor(0x10, 0xB9, 0x81)
ACCENT_DARK = RGBColor(0x06, 0x5F, 0x46)
ACCENT_SOFT = RGBColor(0x1A, 0x3A, 0x30)
DANGER      = RGBColor(0xDC, 0x2E, 0x2E)
DANGER_SOFT = RGBColor(0x4A, 0x1A, 0x1A)
WARN        = RGBColor(0xE0, 0x7E, 0x1C)
TRUNK       = RGBColor(0x5B, 0x3A, 0x22)
STUMP       = RGBColor(0x3A, 0x24, 0x14)

# margins
MARGIN_X = Inches(0.7)
CONTENT_W = W - MARGIN_X * 2

FONT_SANS = 'Inter'
FONT_MONO = 'JetBrains Mono'

# ═══ HELPERS ═══
def new_prs():
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def fill_bg(slide, color=BG):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    return bg

def set_alpha(shape, alpha_pct):
    sp = shape.fill.fore_color._xFill
    srgb = sp.find(qn('a:srgbClr'))
    if srgb is not None:
        for a in srgb.findall(qn('a:alpha')):
            srgb.remove(a)
        alpha = etree.SubElement(srgb, qn('a:alpha'))
        alpha.set('val', str(int(alpha_pct * 1000)))

def add_image_bg(slide, img_path, darken=0.55, crop=None):
    if crop:
        pic = slide.shapes.add_picture(str(img_path), *crop)
    else:
        pic = slide.shapes.add_picture(str(img_path), 0, 0, width=W, height=H)
    overlay = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    overlay.fill.solid(); overlay.fill.fore_color.rgb = BG_DEEP
    overlay.line.fill.background()
    overlay.shadow.inherit = False
    set_alpha(overlay, darken * 100)
    return pic

def add_gradient_overlay(slide, bottom_heavy=True):
    """Vertical dark gradient from bottom to make text pop."""
    if bottom_heavy:
        # stacked semi-transparent bands from bottom up
        bands = 8
        for i in range(bands):
            y = H - Inches(4.5) + Inches(i * 0.56)
            band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, y, W, Inches(0.6))
            band.fill.solid(); band.fill.fore_color.rgb = BG_DEEP
            band.line.fill.background(); band.shadow.inherit = False
            set_alpha(band, (bands - i) * 8)
    else:
        band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
        band.fill.solid(); band.fill.fore_color.rgb = BG_DEEP
        band.line.fill.background(); band.shadow.inherit = False
        set_alpha(band, 30)

def add_text(slide, x, y, w, h, text, *, size=14, bold=False, color=TEXT,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT_SANS,
             spacing=1.2):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split('\n')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb

def rect(slide, x, y, w, h, fill=PANEL, line=None, line_w=0.75):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s

def round_rect(slide, x, y, w, h, fill=PANEL, line=None, line_w=0.75, radius=0.04):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    s.adjustments[0] = radius
    return s

def oval(slide, x, y, w, h, fill=ACCENT, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    return s

def accent_bar(slide, x, y, w=Inches(0.4), h=Inches(0.04), color=ACCENT):
    return rect(slide, x, y, w, h, fill=color)

# ═══ TREE SHAPES ═══
def draw_tree(slide, cx, base_y, height=Inches(1.3), color=ACCENT_DARK, crown_color=None):
    if crown_color is None:
        crown_color = color
    trunk_h = height * 0.18
    crown_h = height - trunk_h
    trunk_w = height * 0.07
    crown_w = height * 0.52

    trunk = rect(slide, cx - trunk_w/2, base_y - trunk_h, trunk_w, trunk_h,
                 fill=TRUNK, line=None)

    t1 = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                cx - crown_w/2, base_y - trunk_h - crown_h,
                                crown_w, crown_h * 0.7)
    t1.fill.solid(); t1.fill.fore_color.rgb = crown_color
    t1.line.fill.background(); t1.shadow.inherit = False

    t2 = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                cx - crown_w*0.42, base_y - trunk_h - crown_h * 0.45,
                                crown_w*0.84, crown_h * 0.6)
    t2.fill.solid(); t2.fill.fore_color.rgb = color
    t2.line.fill.background(); t2.shadow.inherit = False
    return [trunk, t1, t2]

def draw_stump(slide, cx, base_y, height=Inches(1.3)):
    stump_w = height * 0.14
    stump_h = height * 0.14
    shadow = oval(slide, cx - stump_w*1.2, base_y - Inches(0.04), stump_w*2.4, Inches(0.08), fill=BG_DEEP)
    set_alpha(shadow, 40)
    stump = rect(slide, cx - stump_w/2, base_y - stump_h, stump_w, stump_h,
                 fill=STUMP, line=None)
    top = oval(slide, cx - stump_w/2, base_y - stump_h - Inches(0.02),
               stump_w, Inches(0.05), fill=RGBColor(0x8A, 0x5A, 0x38))
    return [shadow, stump, top]

# ═══ TEMPLATE ═══
def slide_header(slide, eyebrow, title, subtitle=None):
    accent_bar(slide, MARGIN_X, Inches(0.42), w=Inches(0.4))
    add_text(slide, MARGIN_X + Inches(0.55), Inches(0.35), Inches(8), Inches(0.3),
             eyebrow.upper(), size=10, color=ACCENT, bold=True, font=FONT_MONO,
             anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, MARGIN_X, Inches(0.95), CONTENT_W, Inches(0.85),
             title, size=34, color=TEXT, bold=True, spacing=1.0)
    if subtitle:
        add_text(slide, MARGIN_X, Inches(1.75), CONTENT_W, Inches(0.4),
                 subtitle, size=15, color=MUTED, spacing=1.3)
    rect(slide, MARGIN_X, Inches(2.35), CONTENT_W, Inches(0.01), fill=BORDER)

def page_number(slide, n, total):
    add_text(slide, Inches(12.55), Inches(7.15), Inches(0.7), Inches(0.3),
             f"{n:02d}—{total:02d}", size=9, color=DIM, align=PP_ALIGN.RIGHT, font=FONT_MONO)
    add_text(slide, MARGIN_X, Inches(7.15), Inches(6), Inches(0.3),
             "GUMP  ·  OSAPIENS MAKEATHON 2026", size=9, color=DIM, font=FONT_MONO)

def add_transition(slide, kind='fade', speed='med'):
    for t in slide.element.findall(qn('p:transition')):
        slide.element.remove(t)
    xml = f'<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="{speed}"><p:{kind}/></p:transition>'
    slide.element.append(etree.fromstring(xml))

# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════
prs = new_prs()
TOTAL = 10

# ───────────────────────────────────────────────
# SLIDE 1 — COVER · pure image, minimal text
# ───────────────────────────────────────────────
s = blank(prs)
add_image_bg(s, IMG / "hero_forest.jpg", darken=0.15)

# soft dark gradient only at bottom for text legibility
for i in range(6):
    y = H - Inches(3.5) + Inches(i * 0.58)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, y, W, Inches(0.65))
    band.fill.solid(); band.fill.fore_color.rgb = BG_DEEP
    band.line.fill.background(); band.shadow.inherit = False
    set_alpha(band, (6 - i) * 10)

# eyebrow — small mono, bottom-left
add_text(s, MARGIN_X, Inches(5.15), Inches(10), Inches(0.3),
         "GUMP  ·  DEFORESTATION DETECTION",
         size=10, color=ACCENT, bold=True, font=FONT_MONO)

# title — two lines, large, bottom-left
add_text(s, MARGIN_X, Inches(5.55), Inches(12), Inches(0.9),
         "The lungs of our planet",
         size=52, color=TEXT, bold=True, spacing=1.0)
add_text(s, MARGIN_X, Inches(6.25), Inches(12), Inches(0.9),
         "are running out of breath.",
         size=52, color=ACCENT, bold=True, spacing=1.0)

# team credit — bottom right, tiny
add_text(s, Inches(9.5), Inches(7.2), Inches(3.5), Inches(0.25),
         "Team Gump  ·  osapiens Makeathon 2026",
         size=9, color=DIM, font=FONT_MONO, align=PP_ALIGN.RIGHT)

add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 2 — THE HOOK · radically simpler
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG_DEEP)

# top brand strip
accent_bar(s, MARGIN_X, Inches(0.42), w=Inches(0.4))
add_text(s, MARGIN_X + Inches(0.55), Inches(0.35), Inches(6), Inches(0.3),
         "THE HOOK", size=10, color=ACCENT, bold=True, font=FONT_MONO,
         anchor=MSO_ANCHOR.MIDDLE)

# one massive question, centered
add_text(s, MARGIN_X, Inches(1.1), CONTENT_W, Inches(0.9),
         "Watch this slide for 10 seconds.",
         size=36, color=TEXT, bold=True, align=PP_ALIGN.CENTER, spacing=1.0)
add_text(s, MARGIN_X, Inches(1.85), CONTENT_W, Inches(0.5),
         "That is all it takes to lose a forest the size of six football fields.",
         size=16, color=MUTED, align=PP_ALIGN.CENTER)

# BIG stat, centered hero
add_text(s, MARGIN_X, Inches(2.7), CONTENT_W, Inches(1.8),
         "4.2 ha",
         size=180, color=ACCENT, bold=True, align=PP_ALIGN.CENTER, spacing=0.9)
add_text(s, MARGIN_X, Inches(4.7), CONTENT_W, Inches(0.35),
         "GONE · EVERY 10 SECONDS · WORLDWIDE",
         size=13, color=TEXT, bold=True, align=PP_ALIGN.CENTER, font=FONT_MONO)

# Before / After — one clean visual row of 10 trees
row_y = Inches(6.5)
n_trees = 10
col_spacing = Inches(1.1)
tree_h = Inches(1.1)
# Center the row of trees
total_row_w = col_spacing * (n_trees - 1)
start_x = (W - total_row_w) / 2

for i in range(n_trees):
    cx = start_x + col_spacing * i
    if i < 6:
        draw_stump(s, cx, row_y, height=tree_h)
    else:
        draw_tree(s, cx, row_y, height=tree_h)

# clean labels below, aligned to two halves
divider_x = start_x + col_spacing * 5.5
rect(s, divider_x, row_y - Inches(1.4), Inches(0.01), Inches(1.5), fill=BORDER_HI)

add_text(s, MARGIN_X, Inches(6.7), Inches(6), Inches(0.3),
         "60 % · DESTROYED", size=11, color=DANGER, bold=True,
         align=PP_ALIGN.CENTER, font=FONT_MONO)
add_text(s, Inches(6.67), Inches(6.7), Inches(6), Inches(0.3),
         "40 % · REMAINING", size=11, color=ACCENT, bold=True,
         align=PP_ALIGN.CENTER, font=FONT_MONO)

page_number(s, 2, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 3 — DATA MODALITIES
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG)

slide_header(s,
    eyebrow="Data",
    title="Three eyes watching the forest",
    subtitle="Multimodal satellite time-series 2020 – 2024, 21 regions, 2.3 GB feature cache."
)

cols = [
    ("01", "SENTINEL-1", "Radar",
     "VV polarization\n1 band  ·  monthly",
     "Sees through clouds and night."),
    ("02", "SENTINEL-2", "Optical (L2A)",
     "12 spectral bands\n10 m  ·  monthly",
     "Color, NDVI, NBR — the visible signal."),
    ("03", "ALPHAEARTH", "Embeddings",
     "64 dims / pixel\nyearly aggregated",
     "Google's pre-trained Earth representation."),
]
col_w = Inches(3.95)
col_h = Inches(4.3)
gap_x = Inches(0.15)
start_x = MARGIN_X
y = Inches(2.7)

for i, (num, title, kind, spec, body) in enumerate(cols):
    x = start_x + (col_w + gap_x) * i
    round_rect(s, x, y, col_w, col_h, fill=PANEL, line=BORDER, line_w=0.75, radius=0.02)
    add_text(s, x + Inches(0.35), y + Inches(0.35), Inches(1), Inches(0.4),
             num, size=11, color=ACCENT, bold=True, font=FONT_MONO)
    accent_bar(s, x + col_w - Inches(0.75), y + Inches(0.4), w=Inches(0.4), h=Inches(0.04))
    add_text(s, x + Inches(0.35), y + Inches(0.85), col_w - Inches(0.7), Inches(0.6),
             title, size=22, color=TEXT, bold=True)
    add_text(s, x + Inches(0.35), y + Inches(1.4), col_w - Inches(0.7), Inches(0.35),
             kind, size=13, color=MUTED)
    rect(s, x + Inches(0.35), y + Inches(1.95), col_w - Inches(0.7), Inches(0.01), fill=BORDER)
    add_text(s, x + Inches(0.35), y + Inches(2.15), col_w - Inches(0.7), Inches(0.8),
             spec, size=12, color=MUTED, font=FONT_MONO, spacing=1.4)
    add_text(s, x + Inches(0.35), y + Inches(3.3), col_w - Inches(0.7), Inches(0.8),
             body, size=13, color=TEXT, spacing=1.4)

page_number(s, 3, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 4 — LABEL NOISE PROBLEM
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG)

slide_header(s,
    eyebrow="The Problem",
    title="The ground truth is noisy",
    subtitle="Three label sources tell us where the forest was cut — none of them agrees 100 %."
)

# Left: agreement timeline
vis_x = MARGIN_X
vis_y = Inches(2.8)
vis_w = Inches(5.8)
vis_h = Inches(4.0)
round_rect(s, vis_x, vis_y, vis_w, vis_h, fill=PANEL, line=BORDER, radius=0.02)
add_text(s, vis_x + Inches(0.3), vis_y + Inches(0.25), vis_w - Inches(0.6), Inches(0.3),
         "ONE PIXEL · THREE SOURCES · TWELVE MONTHS", size=10, color=ACCENT, bold=True, font=FONT_MONO)

track_y = vis_y + Inches(0.8)
track_h = Inches(0.5)
track_w = vis_w - Inches(0.8)
track_x = vis_x + Inches(0.4)
gap = Inches(0.35)

sources = [
    ("RADD",    DANGER, [1,1,0,1,1,1,0,1,1,1,1,0]),
    ("GLAD-L",  WARN,   [0,1,1,1,1,0,0,1,1,1,0,1]),
    ("GLAD-S2", ACCENT, [1,1,1,1,1,1,1,1,1,1,1,1]),
]
n_months = 12
for i, (name, col, pattern) in enumerate(sources):
    ty = track_y + (track_h + gap) * i
    add_text(s, track_x - Inches(0.05), ty, Inches(0.9), track_h, name,
             size=10, color=col, bold=True, font=FONT_MONO, anchor=MSO_ANCHOR.MIDDLE)
    label_offset = Inches(0.9)
    cells_w = track_w - label_offset
    cw = cells_w / n_months
    for j, v in enumerate(pattern):
        cx = track_x + label_offset + cw * j
        c = rect(s, cx, ty, cw - Inches(0.03), track_h,
                 fill=col if v else PANEL_HI, line=BORDER, line_w=0.4)
        if not v:
            set_alpha(c, 30)

# consensus track
ty = track_y + (track_h + gap) * 3 + Inches(0.1)
add_text(s, track_x - Inches(0.05), ty, Inches(0.9), track_h, "≥ 2/3",
         size=10, color=TEXT, bold=True, font=FONT_MONO, anchor=MSO_ANCHOR.MIDDLE)
label_offset = Inches(0.9)
cells_w = track_w - label_offset
cw = cells_w / n_months
for j in range(n_months):
    votes = sum(sources[k][2][j] for k in range(3))
    is_consensus = votes >= 2
    cx = track_x + label_offset + cw * j
    rect(s, cx, ty, cw - Inches(0.03), track_h,
         fill=ACCENT if is_consensus else BG_DEEP,
         line=ACCENT if is_consensus else BORDER, line_w=0.6)

add_text(s, vis_x + Inches(0.4), vis_y + Inches(3.55), vis_w - Inches(0.8), Inches(0.3),
         "Months →", size=9, color=DIM, font=FONT_MONO)

# Right: what goes wrong
list_x = Inches(6.85)
list_y = Inches(2.8)
list_w = Inches(5.8)
list_h = Inches(4.0)
round_rect(s, list_x, list_y, list_w, list_h, fill=PANEL, line=BORDER, radius=0.02)
add_text(s, list_x + Inches(0.3), list_y + Inches(0.25), list_w - Inches(0.6), Inches(0.3),
         "WHAT GOES WRONG", size=10, color=DANGER, bold=True, font=FONT_MONO)

bullets = [
    ("False positives",
     "GLAD-L often flags 'probable' when nothing was actually cut."),
    ("False negatives",
     "RADD misses narrow clearings smaller than 0.5 ha."),
    ("Time lag",
     "GLAD-S2 only starts in 2019 — older events are missing entirely."),
    ("Spatial misalignment",
     "Edge pixels disagree across sources by up to 30 m."),
]
y0 = list_y + Inches(0.8)
for title, desc in bullets:
    oval(s, list_x + Inches(0.35), y0 + Inches(0.08), Inches(0.1), Inches(0.1), fill=DANGER)
    add_text(s, list_x + Inches(0.55), y0, list_w - Inches(0.8), Inches(0.3),
             title, size=13, color=TEXT, bold=True)
    add_text(s, list_x + Inches(0.55), y0 + Inches(0.32), list_w - Inches(0.8), Inches(0.5),
             desc, size=11, color=MUTED, spacing=1.4)
    y0 += Inches(0.78)

page_number(s, 4, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 5 — SOLUTION: CONSENSUS + WEIGHTS
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG)

slide_header(s,
    eyebrow="The Solution",
    title="Consensus beats blind trust",
    subtitle="A pixel is only labeled 'deforested' when ≥ 2 of 3 sources agree. Uncertain pixels count less."
)

form_x = MARGIN_X
form_y = Inches(2.8)
form_w = Inches(5.8)
form_h = Inches(4.1)
round_rect(s, form_x, form_y, form_w, form_h, fill=PANEL, line=BORDER, radius=0.02)
add_text(s, form_x + Inches(0.3), form_y + Inches(0.3), form_w - Inches(0.6), Inches(0.3),
         "CONSENSUS RULE", size=10, color=ACCENT, bold=True, font=FONT_MONO)
add_text(s, form_x + Inches(0.3), form_y + Inches(0.8), form_w - Inches(0.6), Inches(1.0),
         "≥ 2 / 3", size=72, color=TEXT, bold=True, align=PP_ALIGN.CENTER, spacing=1.0)
add_text(s, form_x + Inches(0.3), form_y + Inches(2.0), form_w - Inches(0.6), Inches(0.4),
         "sources must agree",
         size=14, color=MUTED, align=PP_ALIGN.CENTER)

add_text(s, form_x + Inches(0.3), form_y + Inches(2.7), form_w - Inches(0.6), Inches(0.3),
         "PER-PIXEL WEIGHT", size=10, color=ACCENT, bold=True, font=FONT_MONO)
weights = [("3 votes", "1.00", ACCENT),
           ("2 votes", "0.80", WARN),
           ("1 vote",  "0.30", DANGER)]
wy = form_y + Inches(3.1)
for i, (lbl, val, col) in enumerate(weights):
    wx = form_x + Inches(0.3) + Inches(i * 1.78)
    round_rect(s, wx, wy, Inches(1.6), Inches(0.8),
                      fill=BG_DEEP, line=col, line_w=1, radius=0.05)
    add_text(s, wx, wy + Inches(0.08), Inches(1.6), Inches(0.25),
             lbl, size=9, color=MUTED, align=PP_ALIGN.CENTER, font=FONT_MONO)
    add_text(s, wx, wy + Inches(0.3), Inches(1.6), Inches(0.45),
             val, size=22, color=col, bold=True, align=PP_ALIGN.CENTER, font=FONT_MONO)

out_x = Inches(6.85)
out_y = Inches(2.8)
out_w = Inches(5.8)
out_h = Inches(4.1)
round_rect(s, out_x, out_y, out_w, out_h, fill=PANEL, line=ACCENT, line_w=1.5, radius=0.02)
add_text(s, out_x + Inches(0.3), out_y + Inches(0.3), out_w - Inches(0.6), Inches(0.3),
         "WHY IT WORKS", size=10, color=ACCENT, bold=True, font=FONT_MONO)
add_text(s, out_x + Inches(0.3), out_y + Inches(0.75), out_w - Inches(0.6), Inches(0.5),
         "Weighted Focal Loss",
         size=22, color=TEXT, bold=True)

benefits = [
    "Clear-positive pixels dominate training.",
    "Uncertain pixels barely push the gradient.",
    "A single bad source is cancelled out by the other two.",
    "Focal loss prioritizes hard examples — no easy forests.",
]
by = out_y + Inches(1.6)
for b in benefits:
    oval(s, out_x + Inches(0.35), by + Inches(0.08), Inches(0.09), Inches(0.09), fill=ACCENT)
    add_text(s, out_x + Inches(0.55), by, out_w - Inches(0.85), Inches(0.55),
             b, size=12, color=TEXT, spacing=1.4)
    by += Inches(0.55)

add_text(s, MARGIN_X, Inches(7.0), CONTENT_W, Inches(0.3),
         "+ Forest-2020 mask (NDVI > 0.6)     +     Region-held-out validation split",
         size=11, color=MUTED, align=PP_ALIGN.CENTER, font=FONT_MONO)

page_number(s, 5, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 6 — MODEL ARCHITECTURE
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG)

slide_header(s,
    eyebrow="The Model",
    title="Small. Fast. Robust.",
    subtitle="Residual MLP on ~82 features per pixel. Trains in minutes on an AMD MI300X."
)

arch_y = Inches(2.85)
arch_h = Inches(1.8)
segs = [
    ("INPUT",    "82 features", "S2·S1·AEF + Δ"),
    ("RES-MLP",  "256→128→64",  "2× residual"),
    ("DROPOUT",  "p = 0.2",     "BatchNorm"),
    ("SIGMOID",  "Probability", "p ∈ [0, 1]"),
    ("OUTPUT",   "Binary",      "deforested?"),
]
seg_w = Inches(2.3)
seg_gap = Inches(0.2)
x0 = MARGIN_X + Inches(0.15)
for i, (title, main, sub) in enumerate(segs):
    xx = x0 + (seg_w + seg_gap) * i
    is_core = i == 1
    round_rect(s, xx, arch_y, seg_w, arch_h,
                     fill=PANEL_HI if is_core else PANEL,
                     line=ACCENT if is_core else BORDER,
                     line_w=1.5 if is_core else 0.75, radius=0.03)
    add_text(s, xx, arch_y + Inches(0.3), seg_w, Inches(0.3),
             title, size=10, color=ACCENT if is_core else MUTED, bold=True,
             align=PP_ALIGN.CENTER, font=FONT_MONO)
    add_text(s, xx, arch_y + Inches(0.7), seg_w, Inches(0.5),
             main, size=18, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, xx, arch_y + Inches(1.2), seg_w, Inches(0.4),
             sub, size=11, color=MUTED, align=PP_ALIGN.CENTER, font=FONT_MONO)
    if i < len(segs) - 1:
        add_text(s, xx + seg_w, arch_y + Inches(0.6), seg_gap, Inches(0.6),
                 "→", size=18, color=ACCENT, bold=True, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)

tricks_y = Inches(5.15)
tricks = [
    ("FOCAL γ = 2", "Hard examples get higher weight"),
    ("bfloat16 AMP", "Hardware-optimized on MI300X"),
    ("COSINE LR",   "3e-4 with warmup, smooth decay"),
    ("EARLY STOP",  "Best Val-F1 tracked per region"),
]
tw = Inches(2.95)
tg = Inches(0.15)
tx = MARGIN_X
for i, (title, desc) in enumerate(tricks):
    xx = tx + (tw + tg) * i
    round_rect(s, xx, tricks_y, tw, Inches(1.5), fill=PANEL, line=BORDER, radius=0.03)
    rect(s, xx, tricks_y, Inches(0.06), Inches(1.5), fill=ACCENT)
    add_text(s, xx + Inches(0.25), tricks_y + Inches(0.25), tw - Inches(0.5), Inches(0.35),
             title, size=11, color=ACCENT, bold=True, font=FONT_MONO)
    add_text(s, xx + Inches(0.25), tricks_y + Inches(0.7), tw - Inches(0.5), Inches(0.7),
             desc, size=12, color=TEXT, spacing=1.4)

page_number(s, 6, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 7 — GENERALIZATION · two tight, aligned columns
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG)

slide_header(s,
    eyebrow="Generalization",
    title="Why the model works on new regions",
    subtitle="The most common mistake: random pixel splits. A pixel from the same tile must never appear in both train and val."
)

col_h = Inches(4.3)
col_w = Inches(5.85)
col_y = Inches(2.75)

# ═══ LEFT: Naive ═══
lx = MARGIN_X
round_rect(s, lx, col_y, col_w, col_h, fill=PANEL, line=DANGER, line_w=1.2, radius=0.02)
add_text(s, lx + Inches(0.3), col_y + Inches(0.3), col_w - Inches(0.6), Inches(0.3),
         "NAIVE APPROACH", size=10, color=DANGER, bold=True, font=FONT_MONO)
add_text(s, lx + Inches(0.3), col_y + Inches(0.7), col_w - Inches(0.6), Inches(0.5),
         "Random pixel split",
         size=22, color=TEXT, bold=True)

# Visual: one 6x6 grid with mixed train/val dots — centered inside left half
grid_sz = Inches(1.8)
grid_cx = lx + col_w / 2 - Inches(1.7)  # shift grid slightly left
grid_x = grid_cx - grid_sz / 2
grid_y = col_y + Inches(1.7)
cells = 6
cell_sz = grid_sz / cells
import random
random.seed(42)
for r in range(cells):
    for c in range(cells):
        cx = grid_x + c * cell_sz
        cy = grid_y + r * cell_sz
        is_val = random.random() < 0.3
        col = DANGER if is_val else DIM
        rect(s, cx + Inches(0.02), cy + Inches(0.02), cell_sz - Inches(0.04), cell_sz - Inches(0.04),
             fill=col, line=None)
add_text(s, grid_x, grid_y + grid_sz + Inches(0.1), grid_sz, Inches(0.25),
         "ONE TILE · MIXED", size=8, color=DANGER, bold=True,
         align=PP_ALIGN.CENTER, font=FONT_MONO)

# bullets to the right of grid
bx = grid_x + grid_sz + Inches(0.3)
bw = col_w - (bx - lx) - Inches(0.3)
naive_points = [
    "Val pixels are neighbors of train pixels",
    "Model learns tile-specific noise",
    "Val-F1 ≈ 0.9, leaderboard < 0.4",
    "Tests memorization, not generalization",
]
py = col_y + Inches(1.7)
for p in naive_points:
    rect(s, bx, py + Inches(0.15), Inches(0.12), Inches(0.02), fill=DANGER)
    add_text(s, bx + Inches(0.22), py, bw, Inches(0.5),
             p, size=11, color=MUTED, spacing=1.3)
    py += Inches(0.55)

# ═══ RIGHT: Ours ═══
rx = MARGIN_X + col_w + Inches(0.15)
round_rect(s, rx, col_y, col_w, col_h, fill=PANEL, line=ACCENT, line_w=1.2, radius=0.02)
add_text(s, rx + Inches(0.3), col_y + Inches(0.3), col_w - Inches(0.6), Inches(0.3),
         "OUR APPROACH", size=10, color=ACCENT, bold=True, font=FONT_MONO)
add_text(s, rx + Inches(0.3), col_y + Inches(0.7), col_w - Inches(0.6), Inches(0.5),
         "Region-held-out split",
         size=22, color=TEXT, bold=True)

# Visual: two aligned tiles (train) + gap + one tile (val), same grid_y as left
tile_sz = Inches(1.1)
visual_total_w = tile_sz * 2 + Inches(0.25) + Inches(0.25) + tile_sz  # 2 train + gap + 1 val
visual_x0 = rx + Inches(0.35)
visual_y = col_y + Inches(1.7)

# TRAIN tile 1
for r in range(4):
    for c in range(4):
        rect(s, visual_x0 + c*(tile_sz/4), visual_y + r*(tile_sz/4),
             tile_sz/4 - Inches(0.02), tile_sz/4 - Inches(0.02),
             fill=ACCENT_DARK, line=None)
add_text(s, visual_x0, visual_y + tile_sz + Inches(0.1), tile_sz, Inches(0.25),
         "TILE A · TRAIN", size=8, color=ACCENT_DARK, bold=True, align=PP_ALIGN.CENTER, font=FONT_MONO)

# TRAIN tile 2 (right next to tile 1)
t2x = visual_x0 + tile_sz + Inches(0.1)
for r in range(4):
    for c in range(4):
        rect(s, t2x + c*(tile_sz/4), visual_y + r*(tile_sz/4),
             tile_sz/4 - Inches(0.02), tile_sz/4 - Inches(0.02),
             fill=ACCENT_DARK, line=None)
add_text(s, t2x, visual_y + tile_sz + Inches(0.1), tile_sz, Inches(0.25),
         "TILE B · TRAIN", size=8, color=ACCENT_DARK, bold=True, align=PP_ALIGN.CENTER, font=FONT_MONO)

# gap line
gap_line_x = t2x + tile_sz + Inches(0.12)
rect(s, gap_line_x, visual_y - Inches(0.1), Inches(0.02), tile_sz + Inches(0.4), fill=ACCENT)

# VAL tile (separate!)
v3x = gap_line_x + Inches(0.15)
for r in range(4):
    for c in range(4):
        rect(s, v3x + c*(tile_sz/4), visual_y + r*(tile_sz/4),
             tile_sz/4 - Inches(0.02), tile_sz/4 - Inches(0.02),
             fill=ACCENT, line=None)
add_text(s, v3x, visual_y + tile_sz + Inches(0.1), tile_sz, Inches(0.25),
         "TILE 48P · VAL", size=8, color=ACCENT, bold=True, align=PP_ALIGN.CENTER, font=FONT_MONO)

# bullets below tiles
our_points = [
    "Val region completely untouched in training",
    "Not a single pixel leaks between sets",
    "Val-F1 honestly approximates leaderboard",
    "Forces real pattern learning, not memorization",
]
py = col_y + Inches(3.05)
for p in our_points:
    rect(s, rx + Inches(0.3), py + Inches(0.15), Inches(0.12), Inches(0.02), fill=ACCENT)
    add_text(s, rx + Inches(0.45), py, col_w - Inches(0.75), Inches(0.35),
             p, size=11, color=TEXT, spacing=1.3)
    py += Inches(0.32)

page_number(s, 7, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 8 — RESULTS
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG_DEEP)

slide_header(s,
    eyebrow="Results",
    title="Baseline model on region 48P",
    subtitle="Held-out validation. Threshold calibrated on Val-F1."
)

hero_x = MARGIN_X
hero_y = Inches(2.85)
hero_w = Inches(5.8)
hero_h = Inches(4.0)
round_rect(s, hero_x, hero_y, hero_w, hero_h, fill=BG, line=ACCENT, line_w=1.5, radius=0.02)
add_text(s, hero_x + Inches(0.4), hero_y + Inches(0.35), hero_w - Inches(0.8), Inches(0.3),
         "F1-SCORE", size=10, color=ACCENT, bold=True, font=FONT_MONO)
add_text(s, hero_x + Inches(0.4), hero_y + Inches(0.85), hero_w - Inches(0.8), Inches(2.2),
         "0.554", size=130, color=TEXT, bold=True, align=PP_ALIGN.CENTER, spacing=1.0)

bar_y = hero_y + Inches(2.9)
bar_h = Inches(0.25)
bar_full_w = hero_w - Inches(0.8)
rect(s, hero_x + Inches(0.4), bar_y, bar_full_w, bar_h, fill=PANEL, line=None)
rect(s, hero_x + Inches(0.4), bar_y, bar_full_w * 0.554, bar_h, fill=ACCENT, line=None)
add_text(s, hero_x + Inches(0.4), bar_y + Inches(0.35), hero_w - Inches(0.8), Inches(0.3),
         "Balanced · robust to class imbalance",
         size=12, color=MUTED, align=PP_ALIGN.CENTER)

rx = Inches(6.85)
metrics = [
    ("PRECISION", "0.386", "Of all 'deforested' predictions, ~39 % are correct."),
    ("RECALL",    "0.980", "98 % of real clearings are caught by the model."),
    ("THRESHOLD", "0.90",  "Conservatively calibrated — not blindly 0.5."),
]
for i, (title, val, desc) in enumerate(metrics):
    y = Inches(2.85 + i * 1.35)
    round_rect(s, rx, y, Inches(5.8), Inches(1.2), fill=PANEL, line=BORDER, line_w=0.75, radius=0.03)
    rect(s, rx, y, Inches(0.06), Inches(1.2), fill=ACCENT)
    add_text(s, rx + Inches(0.3), y + Inches(0.15), Inches(3), Inches(0.3),
             title, size=10, color=ACCENT, bold=True, font=FONT_MONO)
    add_text(s, rx + Inches(0.3), y + Inches(0.45), Inches(1.8), Inches(0.7),
             val, size=32, color=TEXT, bold=True)
    add_text(s, rx + Inches(2.1), y + Inches(0.4), Inches(3.5), Inches(0.8),
             desc, size=11, color=MUTED, spacing=1.4)

page_number(s, 8, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 9 — TOOL / DEMO
# ───────────────────────────────────────────────
s = blank(prs)
fill_bg(s, BG)

slide_header(s,
    eyebrow="The Tool",
    title="Gump — an interactive map for regulators",
    subtitle="Local Next.js app. Loads GeoJSON, filters, exports. No login, no cloud."
)

feats = [
    ("LIVE MAP",         "21 tiles · 95 polygons",
     "Leaflet + Esri satellite layer. Pan, zoom, click."),
    ("MODEL METRICS",    "F1 / Precision / Recall",
     "Per region, live from metrics.json — no hardcoded numbers."),
    ("CONFIDENCE MODE",  "Color gradient",
     "Red = uncertain · yellow = medium · green = high confidence."),
    ("SEARCH & FILTER",  "Year · area · confidence",
     "Interactive sliders. Results update instantly."),
    ("EXPORT",           "CSV · merged GeoJSON",
     "One click for leaderboard upload. EPSG:4326 guaranteed."),
    ("OFFLINE",          "100 % local",
     "Runs without internet. No privacy overhead."),
]
col_w = Inches(3.95)
col_h = Inches(1.95)
gap_x = Inches(0.15)
gap_y = Inches(0.2)
start_x = MARGIN_X
start_y = Inches(2.75)
for i, (title, tag, desc) in enumerate(feats):
    row = i // 3
    col = i % 3
    x = start_x + (col_w + gap_x) * col
    y = start_y + (col_h + gap_y) * row
    round_rect(s, x, y, col_w, col_h, fill=PANEL, line=BORDER, line_w=0.75, radius=0.03)
    rect(s, x, y, Inches(0.06), col_h, fill=ACCENT)
    add_text(s, x + Inches(0.25), y + Inches(0.2), col_w - Inches(0.5), Inches(0.3),
             title, size=10, color=ACCENT, bold=True, font=FONT_MONO)
    add_text(s, x + Inches(0.25), y + Inches(0.55), col_w - Inches(0.5), Inches(0.4),
             tag, size=15, color=TEXT, bold=True)
    add_text(s, x + Inches(0.25), y + Inches(1.05), col_w - Inches(0.5), Inches(0.8),
             desc, size=11, color=MUTED, spacing=1.4)

add_text(s, MARGIN_X, Inches(6.95), CONTENT_W, Inches(0.3),
         "localhost:8765  ·  npm run dev",
         size=11, color=ACCENT, bold=True, align=PP_ALIGN.CENTER, font=FONT_MONO)

page_number(s, 9, TOTAL)
add_transition(s, 'fade')

# ───────────────────────────────────────────────
# SLIDE 10 — CLOSING · pure image, one line
# ───────────────────────────────────────────────
s = blank(prs)
add_image_bg(s, IMG / "tree_canopy.jpg", darken=0.55)

# centered statement — single line, massive
add_text(s, MARGIN_X, Inches(3.0), CONTENT_W, Inches(1.2),
         "Every hectare counts.",
         size=66, color=TEXT, bold=True, align=PP_ALIGN.CENTER, spacing=1.0)

# thin accent line below
rect(s, W/2 - Inches(1.5), Inches(4.35), Inches(3.0), Inches(0.05), fill=ACCENT)

# team credit — bottom center, tiny
add_text(s, MARGIN_X, Inches(7.1), CONTENT_W, Inches(0.3),
         "Team Gump  ·  osapiens Makeathon 2026",
         size=10, color=DIM, font=FONT_MONO, align=PP_ALIGN.CENTER)

add_transition(s, 'fade')

prs.save(OUT)
print(f"✓ {OUT}  ({OUT.stat().st_size // 1024} KB, {len(prs.slides)} slides)")
