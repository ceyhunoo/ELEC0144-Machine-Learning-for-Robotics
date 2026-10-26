# CODE MAP of ThresholdHistograms.py
# Settings        file name, the seven thresholds of Task1.py and their working ranges
# Sample outlines rough outlines of the three fruits, used only to collect sample pixels
# Load image      read Fruits.jpg, compute R, G, B, the ratios and relative brightness
# Pixel groups    pixels inside each outline (one fruit) and pixels outside all outlines (the rest)
# Histograms      one histogram per colour condition: the fruit against the rest, threshold marked
# Pass rates      share of fruit pixels and of other pixels that pass each condition and each full rule
# AI use          Claude Opus 5.5 (claude.ai), default settings
#
# What this file is for: it shows how the thresholds in Task1.py were chosen (Week 1 slide 29:
# use a histogram to place a threshold). It is NOT part of the Task 1 method. The outlines below
# are drawn by hand only to collect example pixels for the histograms; Task1.py contains no
# coordinates and no hand-drawn masks.
# Run from the folder that contains Fruits.jpg:  python ThresholdHistograms.py

# %% Settings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

IMAGE_FILE = 'Fruits.jpg'
# name: (value used in Task1.py, working range found by moving it alone)
THRESHOLDS = {
    'CAPS_B_MAX':       (0.05, (0.005, 0.10)),
    'CAPS_BRIGHT_MIN':  (0.9,  (0.65, 1.15)),
    'APPLE_G_MAX':      (0.6,  (0.3, 0.85)),
    'APPLE_BG_MIN':     (-0.1, (-0.2, 0.1)),
    'APPLE_BRIGHT_MIN': (0.8,  (0.5, 1.2)),
    'LEMON_G_MIN':      (0.75, (0.6, 0.9)),
    'LEMON_B_MAX':      (0.4,  (0.2, 0.6)),
}

# %% Sample outlines
# (x, y) corners of a rough outline around what is visible of each fruit, in the original
# (not mirrored) image. x = column from the left, y = row from the top.
OUTLINES = {
    'capsicum': [(60, 395), (75, 372), (100, 358), (140, 348), (180, 346), (215, 352), (245, 368),
                 (265, 392), (268, 420), (262, 460), (255, 500), (240, 530), (215, 540), (180, 528),
                 (140, 510), (100, 490), (70, 470), (55, 440), (55, 415)],
    'apple':    [(245, 470), (280, 445), (330, 433), (390, 430), (440, 440), (475, 465), (495, 500),
                 (500, 555), (400, 555), (300, 545), (245, 535), (238, 500)],
    'lemon':    [(555, 265), (575, 230), (610, 205), (655, 197), (700, 203), (740, 225), (765, 265),
                 (770, 300), (762, 340), (745, 345), (735, 300), (715, 285), (690, 290), (670, 300),
                 (640, 295), (600, 292), (575, 290), (560, 285)],
}

# %% Load image
img = plt.imread(IMAGE_FILE)
H, W, _ = img.shape
I = img.astype(float)
R, G, B = I[:, :, 0], I[:, :, 1], I[:, :, 2]
Rsafe = np.maximum(R, 1)                    # avoid dividing by 0 where R = 0
grey = (R + G + B) / 3
features = {                                # the quantities the Task1.py rules compare
    'B/R': B / Rsafe,
    'G/R': G / Rsafe,
    '(B-G)/R': (B - G) / Rsafe,
    'bright': grey / grey.mean(),
}

# %% Pixel groups
X, Y = np.meshgrid(np.arange(W), np.arange(H))
points = np.column_stack([X.ravel(), Y.ravel()])
inside = {f: Path(c).contains_points(points).reshape(H, W) for f, c in OUTLINES.items()}
rest = ~(inside['capsicum'] | inside['apple'] | inside['lemon'])

# %% Histograms
# (fruit, feature, threshold name, which side is kept, x-axis range)
PANELS = [
    ('capsicum', 'B/R',     'CAPS_B_MAX',       'below', (0, 1.2)),
    ('capsicum', 'bright',  'CAPS_BRIGHT_MIN',  'above', (0, 2.6)),
    ('apple',    'G/R',     'APPLE_G_MAX',      'below', (0, 1.6)),
    ('apple',    '(B-G)/R', 'APPLE_BG_MIN',     'above', (-1, 1)),
    ('apple',    'bright',  'APPLE_BRIGHT_MIN', 'above', (0, 2.6)),
    ('lemon',    'G/R',     'LEMON_G_MIN',      'above', (0, 1.6)),
    ('lemon',    'B/R',     'LEMON_B_MAX',      'below', (0, 1.2)),
]
COLOURS = {'capsicum': '#e0680f', 'apple': '#cc2940', 'lemon': '#c99a00'}
SLOTS = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for (row, col), (fruit, feat, tname, side, xr) in zip(SLOTS, PANELS):
    ax = axes[row, col]
    value, (lo, hi) = THRESHOLDS[tname]
    bins = np.linspace(xr[0], xr[1], 81)
    v = np.clip(features[feat], xr[0], xr[1])             # out-of-range values go in the end bins
    ax.hist(v[rest], bins=bins, density=True, color='0.6', alpha=0.6, label='rest of the image')
    ax.hist(v[inside[fruit]], bins=bins, density=True, color=COLOURS[fruit], alpha=0.65, label=fruit)
    ax.axvspan(lo, hi, color='tab:blue', alpha=0.10)          # working range (blue band)
    ax.axvline(value, color='k', lw=1.5)                       # value used in Task1.py
    sign = '<' if side == 'below' else '>'
    ax.set_title(f'{fruit}: keep {feat} {sign} {value}   (works {lo} to {hi})', fontsize=10.5)
    ax.set_xlabel(feat)
    ax.set_ylabel('share of pixels (density)')
    ax.set_xlim(xr)
    if fruit == 'capsicum' and feat == 'B/R':                  # its spike at 0 would flatten the rest
        peak = np.histogram(v[inside[fruit]], bins=bins, density=True)[0].max()
        ax.set_ylim(0, 8)
        ax.annotate(f'capsicum bar at 0\nreaches {peak:.0f}', xy=(0.0, 7.6), xytext=(0.25, 6),
                    fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.legend(fontsize=8.5, loc='best', framealpha=0.85)
    ax.grid(alpha=0.3)

# the two free slots: the sample outlines, and a note on reading the plots
ax = axes[0, 2]
ax.imshow(img)
for fruit, corners in OUTLINES.items():
    c = np.array(corners + [corners[0]])
    ax.plot(c[:, 0], c[:, 1], color='white', lw=2)
    ax.text(c[:, 0].min(), c[:, 1].min() - 10, fruit, color='white', fontsize=9, fontweight='bold')
ax.set_title('Rough outlines used to collect sample pixels\n(this file only, not in Task1.py)', fontsize=10)
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
ax = axes[2, 2]
ax.axis('off')
ax.text(0, 1, 'How to read these plots\n\n'
        'Coloured: pixels of that fruit.\nGrey: every pixel outside the three outlines.\n'
        'Black line: the value in Task1.py.\nBlue band: range over which all three fruits\nare still found (moving that value alone).\n'
        'Each histogram is scaled to area 1, so the two\nshapes compare directly.\n\n'
        'A threshold goes where the fruit\'s pixels mostly\nlie on one side and the rest on the other.\n'
        'No single condition removes everything else:\neach rule needs its conditions together\n(see the pass rates printed in the terminal).\n\n'
        'Ratios do not change with darkFactor,\nso one set of histograms covers 0.6 to 1.0.',
        va='top', fontsize=10)
fig.suptitle('How the Task1.py thresholds were placed: colour ratios, each fruit against the rest of Fruits.jpg', fontsize=13)
plt.tight_layout()

# %% Pass rates
def passes(feat, tname, side):
    value = THRESHOLDS[tname][0]
    return features[feat] < value if side == 'below' else features[feat] > value

print(f'{"condition":34s} {"fruit pixels kept":>18s} {"other pixels kept":>18s}')
for fruit in ['capsicum', 'apple', 'lemon']:
    full = np.ones((H, W), dtype=bool)
    for f2, feat, tname, side, _ in PANELS:
        if f2 != fruit:
            continue
        m = passes(feat, tname, side)
        full &= m
        cond = f'{feat} {"<" if side == "below" else ">"} {THRESHOLDS[tname][0]}'
        print(f'{fruit + ": " + cond:34s} {100 * m[inside[fruit]].mean():17.1f}% {100 * m[rest].mean():17.1f}%')
    print(f'{fruit + ": all conditions together":34s} {100 * full[inside[fruit]].mean():17.1f}% '
          f'{100 * full[rest].mean():17.1f}%   ({full[rest].sum()} other pixels)')
    print()
plt.show()
