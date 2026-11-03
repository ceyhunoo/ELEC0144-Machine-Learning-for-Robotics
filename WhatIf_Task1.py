# CODE MAP of WhatIf_Task1.py  (What-if 1: darken the image, keep everything else as submitted)
# Settings        the 19 darkFactor values, 1.0 down to 0.1 in steps of 0.05
# Run Task1       run_task1: runs Task1.py with only its darkFactor line changed; returns boxes and regions
# Run all         every darkFactor; a fruit counts as found when Task1.py returns a box for it
# Compare         region size and overlap (intersection over union) of each box with its box at 1.0
# Print           one row per darkFactor, then the summary used in the report (Table W1)
# Figure          all 19 boxes of each fruit on the full-brightness image; region size against darkFactor
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the folder that contains Task1.py and Fruits.jpg:  python WhatIf_Task1.py
# Task1.py is not edited: its text is read and run with "darkFactor = 0.8" replaced by each value.
# Its thresholds, median filter size and mirror = True stay exactly as submitted.
import contextlib
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # Task1.py opens a figure each run; no windows here
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DARK_FACTORS = np.round(np.arange(1.0, 0.0999, -0.05), 2)     # 1.0, 0.95, ..., 0.1 (19 values)
FRUITS = ['apple', 'capsicum', 'lemon']
TASK1_SOURCE = open('Task1.py').read()
assert TASK1_SOURCE.count('darkFactor = 0.8 ') == 1           # the one line that is changed


# %% Run Task1
def run_task1(dark_factor: float) -> dict:
    """Run Task1.py with darkFactor set to dark_factor. Returns {fruit: (box, region pixels)} for
    every fruit it finds; box = (x_min, x_max, y_min, y_max). Task1.py's own printout is hidden."""
    code = TASK1_SOURCE.replace('darkFactor = 0.8 ', f'darkFactor = {dark_factor} ', 1)
    names = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, 'Task1.py', 'exec'), names)
    plt.close('all')
    return {f: (tuple(int(v) for v in names['results'][f]['box']), int(names['blobs'][f].sum()))
            for f in names['results']}


# %% Run all
runs = {k: run_task1(k) for k in DARK_FACTORS}


# %% Compare
def box_area(b):
    return (b[1] - b[0] + 1) * (b[3] - b[2] + 1)


def iou(a, b):
    """Intersection over union of two boxes: 1 = identical, 0 = no overlap."""
    w = min(a[1], b[1]) - max(a[0], b[0]) + 1
    h = min(a[3], b[3]) - max(a[2], b[2]) + 1
    inter = max(w, 0) * max(h, 0)
    return inter / (box_area(a) + box_area(b) - inter)


reference = runs[1.0]                       # boxes and regions at full brightness


# %% Print
print('darkFactor  ' + '  '.join(f'{f:>22s}' for f in FRUITS))
print(' ' * 12 + '  '.join(f'{"region px  IoU vs 1.0":>22s}' for _ in FRUITS))
for k in DARK_FACTORS:
    cells = []
    for f in FRUITS:
        if f in runs[k]:
            box, px = runs[k][f]
            cells.append(f'{px:>12d}  {iou(box, reference[f][0]):8.3f}')
        else:
            cells.append(f'{"not found":>22s}')
    print(f'{k:10.2f}  ' + '  '.join(cells))

print('\nSummary (Table W1)')
print(f'{"fruit":9s} {"found":>7s} {"first lost at":>14s} {"at 0.10":>8s} {"smallest IoU":>13s} {"region 1.0 -> 0.1":>19s}')
for f in FRUITS:
    found = [k for k in DARK_FACTORS if f in runs[k]]
    lost = [k for k in DARK_FACTORS if f not in runs[k]]
    smallest = min(iou(runs[k][f][0], reference[f][0]) for k in found)
    px0, px1 = reference[f][1], runs[0.1][f][1] if f in runs[0.1] else 0
    print(f'{f:9s} {len(found):>4d}/19 {(f"{max(lost):.2f}" if lost else "none"):>14s} '
          f'{("found" if f in runs[0.1] else "lost"):>8s} {smallest:13.3f} '
          f'{px0:>8d} -> {px1:<6d} ({(px1 - px0) / px0:+.1%})')
print('\nA fruit is lost when it has no box, or its box covers less than half of its visible area.'
      '\nCoverage is checked by eye on the figure; an IoU near 1 means the box is where it was at full brightness.')


# %% Figure
# Sized for the report (about 15 cm wide), so the text prints at about 6-7 pt.
plt.rcParams.update({'font.size': 7})
img = plt.imread('Fruits.jpg')[:, ::-1, :]          # full brightness, mirrored as in Task1.py
colours = {'apple': 'red', 'capsicum': 'darkorange', 'lemon': 'gold'}
markers = {'apple': 'o', 'capsicum': 's', 'lemon': '^'}
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax0.imshow(img)
for f in FRUITS:
    for k in DARK_FACTORS:
        if f in runs[k]:
            x0, x1, y0, y1 = runs[k][f][0]
            ax0.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False,
                                    edgecolor=colours[f], linewidth=0.8, alpha=0.8))
    ax0.plot([], [], color=colours[f], label=f'{f}: 19 boxes')
ax0.set_title('(a) Boxes from all 19 darkFactor settings, full-brightness image', fontsize=7)
ax0.set_xlabel('x (pixels)'); ax0.set_ylabel('y (pixels)'); ax0.legend(loc='lower right', fontsize=6)
for f in FRUITS:
    ax1.plot(DARK_FACTORS, [runs[k][f][1] / reference[f][1] for k in DARK_FACTORS], marker=markers[f],
             color=colours[f] if f != 'lemon' else 'goldenrod', label=f, markersize=3, lw=0.9)
ax1.axhline(1, color='grey', lw=0.6, ls=':')
ax1.invert_xaxis()
ax1.set_xlabel('darkFactor'); ax1.set_ylabel('region size / size at darkFactor 1.0')
ax1.set_title('(b) Selected region of each fruit', fontsize=7); ax1.legend(fontsize=6)
fig.tight_layout()
fig.savefig('WhatIf_Task1.png', dpi=300)
print('Figure saved: WhatIf_Task1.png')
