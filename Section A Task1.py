# CODE MAP of Task1.py
# Settings        darkFactor, mirror, file name, colour thresholds, median filter size
# Functions       median_filter, label_blobs (both written from scratch with NumPy)
# Load image      read Fruits.jpg, darken (x darkFactor, rounded to 8 bits), mirror
# Colour masks    one binary mask per fruit from channel ratios and relative brightness
# Clean masks     median filter on each mask removes isolated pixels and thin streaks
# Pick blobs      label connected regions of each mask, keep the largest one
# Measure         bounding box, box-midpoint centroid, moment centroid, distance between them
# Plot and print  draw each box and both centroids, print the numbers
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the folder that contains Fruits.jpg:  python Task1.py
# Pixel coordinates: x = column (0 at the left), y = row (0 at the top), as NumPy indexes them.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

darkFactor = 0.8          # every pixel is multiplied by this, then rounded to 8 bits (must work 0.6-1.0)
mirror = True             # True: flip the image left-to-right
IMAGE_FILE = 'Fruits.jpg'

# Colour rules. R, G, B are the channel values of one pixel; "bright" is that pixel's
# (R+G+B)/3 divided by the mean of (R+G+B)/3 over the whole image.
# Each value was moved one at a time; the bracket is the range over which all three
# fruits were still found correctly at darkFactor 1.0, 0.8 mirrored and 0.6.
# The chosen values sit near the middle of their ranges.
CAPS_B_MAX       = 0.05   # capsicum: B < 0.05 R          (0.005 - 0.10)
CAPS_BRIGHT_MIN  = 0.9    # capsicum: bright > 0.9        (0.65 - 1.15)
APPLE_G_MAX      = 0.6    # apple:    G < 0.6 R           (0.3 - 0.85)
APPLE_BG_MIN     = -0.1   # apple:    B - G > -0.1 R      (-0.2 - 0.1)
APPLE_BRIGHT_MIN = 0.8    # apple:    bright > 0.8        (0.5 - 1.2)
LEMON_G_MIN      = 0.75   # lemon:    G > 0.75 R          (0.6 - 0.9)
LEMON_B_MAX      = 0.4    # lemon:    B < 0.4 R           (0.2 - 0.6)
MEDIAN_SIZE      = 7      # median filter window, 7 x 7 pixels (1 - 11 give almost the same boxes)


# %% Functions
def median_filter(I: np.ndarray, n: int) -> np.ndarray:
    """Median of the n x n neighbourhood of every pixel (lecture slides 50-54).
    In:  I, 2-D array; n, odd window size.  Out: filtered array, same size as I.
    Border pixels are repeated outwards so the output keeps the input size."""
    H, W = I.shape
    r = n // 2
    P = np.pad(I, r, mode='edge')
    # stack the n*n shifted copies of the image: shifts[k, y, x] is the k-th neighbour of pixel (x, y)
    shifts = np.stack([P[i:i + H, j:j + W] for i in range(n) for j in range(n)])
    return np.median(shifts, axis=0)


def label_blobs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give every connected region (blob) of True pixels its own number (lecture slide 82).
    In:  mask, 2-D boolean array.
    Out: labels, same size, 0 = background, 1, 2, ... = blob number;
         sizes, sizes[b] = number of pixels in blob b (sizes[0] = 0).
    Neighbours are the 8 surrounding pixels. Each blob is grown from one seed pixel
    with a stack: take a pixel off the stack, add its unlabelled mask neighbours."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    sizes = [0]
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if labels[y0, x0] != 0:              # already part of a blob
            continue
        b = len(sizes)                       # number for the new blob
        labels[y0, x0] = b
        stack = [(y0, x0)]
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and labels[yy, xx] == 0:
                        labels[yy, xx] = b
                        stack.append((yy, xx))
        sizes.append(count)
    return labels, np.array(sizes)


# %% Load image
img = plt.imread(IMAGE_FILE)                              # uint8, shape (rows, cols, 3)
img = np.floor(darkFactor * img + 0.5).astype(np.uint8)   # darken, round to nearest (as MATLAB round)
if mirror:
    img = img[:, ::-1, :]                                 # flip left-to-right
H, W, _ = img.shape

I = img.astype(float)       # work in floats so sums do not overflow at 255
R, G, B = I[:, :, 0], I[:, :, 1], I[:, :, 2]
grey = (R + G + B) / 3                     # brightness of each pixel (slide 18)
bright = grey / grey.mean()                # relative brightness: darkFactor cancels out


# %% Colour masks
# Each rule compares channels with R, so multiplying the whole image by darkFactor
# scales both sides equally and the rule gives (almost) the same answer.
masks = {
    'capsicum': (B < CAPS_B_MAX * R) & (bright > CAPS_BRIGHT_MIN),
    'apple':    (G < APPLE_G_MAX * R) & (B - G > APPLE_BG_MIN * R) & (bright > APPLE_BRIGHT_MIN),
    'lemon':    (G > LEMON_G_MIN * R) & (B < LEMON_B_MAX * R),
}


# %% Clean masks
# Median of 0/1 values = 1 when more than half of the window is 1, so isolated
# pixels and streaks thinner than about half the window disappear.
clean = {fruit: median_filter(m.astype(float), MEDIAN_SIZE) > 0.5 for fruit, m in masks.items()}


# %% Pick blobs
blobs = {}
for fruit, m in clean.items():
    labels, sizes = label_blobs(m)
    if sizes.max() == 0:
        print(f'{fruit}: not found')
        continue
    biggest = np.argmax(sizes)               # the fruit is the largest region of its colour
    blobs[fruit] = labels == biggest


# %% Measure
X, Y = np.meshgrid(np.arange(W), np.arange(H))   # X[y, x] = x, Y[y, x] = y
results = {}
for fruit, blob in blobs.items():
    ys, xs = np.nonzero(blob)
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    # (a) midpoint of the bounding box (slide 84)
    xc_box, yc_box = (x_min + x_max) / 2, (y_min + y_max) / 2
    # (b) moments with I(x, y) = 1 on the blob, 0 elsewhere (slides 86-89)
    M00 = np.sum(blob)            # area
    M10 = np.sum(X * blob)        # sum of x over the blob
    M01 = np.sum(Y * blob)        # sum of y over the blob
    xc_mom, yc_mom = M10 / M00, M01 / M00
    dist = np.hypot(xc_box - xc_mom, yc_box - yc_mom)
    results[fruit] = dict(box=(x_min, x_max, y_min, y_max), mid=(xc_box, yc_box),
                          mom=(xc_mom, yc_mom), dist=dist)


# %% Plot and print
colours = {'capsicum': 'orange', 'apple': 'red', 'lemon': 'yellow'}
fig, ax = plt.subplots(figsize=(10, 7.8))
ax.imshow(img)
for fruit, r in results.items():
    x_min, x_max, y_min, y_max = r['box']
    ax.add_patch(Rectangle((x_min - 0.5, y_min - 0.5), x_max - x_min + 1, y_max - y_min + 1,
                           fill=False, edgecolor=colours[fruit], linewidth=2))
    ax.text(x_min, y_min - 6, fruit, color=colours[fruit], fontsize=11, fontweight='bold')
    ax.plot(*r['mid'], 'x', color='white', markersize=10, markeredgewidth=2)
    ax.plot(*r['mom'], 'o', markerfacecolor='none', markeredgecolor='cyan', markersize=10, markeredgewidth=2)
ax.plot([], [], 'x', color='white', markeredgewidth=2, label='centroid: box midpoint')
ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='cyan', markeredgewidth=2, label='centroid: moments')
ax.legend(loc='lower right')
ax.set_title(f'Task 1: darkFactor = {darkFactor}, mirror = {mirror}')
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
plt.tight_layout()

print(f'darkFactor = {darkFactor}, mirror = {mirror}, image {W} x {H} pixels')
print(f'{"fruit":9s} {"box x":>11s} {"box y":>11s} {"box midpoint (x, y)":>21s} '
      f'{"moments (x, y)":>17s} {"distance":>9s}')
for fruit, r in results.items():
    x_min, x_max, y_min, y_max = r['box']
    print(f'{fruit:9s} {x_min:4d}-{x_max:<6d} {y_min:4d}-{y_max:<6d} '
          f'({r["mid"][0]:6.1f}, {r["mid"][1]:6.1f})      ({r["mom"][0]:6.1f}, {r["mom"][1]:6.1f}) '
          f'{r["dist"]:7.1f} px')
plt.show()
