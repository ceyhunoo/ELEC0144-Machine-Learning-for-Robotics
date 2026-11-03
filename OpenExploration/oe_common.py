# CODE MAP of oe_common.py  (shared functions, imported by the OE files and FinalMethod.py)
# Settings        paths, the table of videos (person, label), frame rate and size used everywhere
# Read video      read_frames: decode a video with ffmpeg into a NumPy array, cached on disk
# Hand and object hand_mask (skin colour), cube_visible (green pixels), where the cube clip ends
# Cut into grips  find_grips: one grip = one stretch with the hand in the frame
# Grip features   hand area, object area and shape, lift and motion, appearance; per grip,
#                 relative to the empty scene just before it
# Hand-free frames frames between grips with no hand in view, and coarse colour features of
#                 them (used by the session check: can a take be recognised without the grip?)
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
from pathlib import Path
import subprocess
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent                     # .../OpenExploration
VIDEO_DIR = HERE.parent / 'open exploration videos'        # the team's recordings (not in git)
CACHE_DIR = HERE / 'cache'                                 # decoded frames (not in git, rebuilt if missing)
RESULTS_DIR = HERE / 'results'                             # csv files and figures used in the report

# One video per person per label. "gentle" in the file names is the brief's "light".
VIDEOS = {
    'Ceyhun firm.mp4':             ('Ceyhun',  'firm'),
    'Ceyhun gentle.mp4':           ('Ceyhun',  'light'),
    'Shangze firm.mp4':            ('Shangze', 'firm'),
    'Shangze gentle.mp4':          ('Shangze', 'light'),
    'Yikhui firm.mp4':             ('Yik Hui', 'firm'),
    'Yikhui gentle.mp4':           ('Yik Hui', 'light'),
    '20261029_131159 firm k.mp4':  ('Kairi',   'firm'),
    '20261029_131437 gentle K.mp4': ('Kairi',  'light'),
}
PEOPLE = ['Ceyhun', 'Kairi', 'Shangze', 'Yik Hui']   # Ceyhun Ozturk, Kairi Shukla, Shangze Kong, Yik Hui Ng

FPS = 15                  # frames per second kept from the 30 fps videos
W, H = 320, 180           # frame size kept (the videos are 1920 x 1080, so 1/6 scale)

try:                      # ffmpeg comes with the imageio-ffmpeg package (pip), so no separate install
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = 'ffmpeg'     # otherwise the system's ffmpeg


# %% Read video
def read_frames(video_name: str, fps: int = FPS, w: int = W, h: int = H) -> np.ndarray:
    """All frames of one video as an array (n_frames, h, w, 3), uint8 RGB.
    The first call decodes with ffmpeg and saves a .npy file in cache/; later calls load it."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f'{Path(video_name).stem}_{fps}fps_{w}x{h}.npy'
    if cache.exists():
        return np.load(cache)
    cmd = [FFMPEG, '-v', 'error', '-i', str(VIDEO_DIR / video_name),
           '-vf', f'fps={fps},scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    np.save(cache, frames)
    return frames


# %% Hand and object
# Skin colour in OpenCV HSV (H 0-180, S and V 0-255). The table is grey-white (low S), the cube
# green (H about 60), the sponge yellow (H about 25-30) and dark grey-green, so only skin passes.
SKIN_H_LOW, SKIN_H_HIGH = 20, 165     # skin hue: H <= 20 or H >= 165 (red-orange)
SKIN_S_MIN, SKIN_S_MAX = 35, 200      # skin is moderately saturated
SKIN_V_MIN = 50                       # ignore very dark pixels


def hand_mask(frames: np.ndarray) -> np.ndarray:
    """Boolean mask (n, h, w): True where a pixel has skin colour."""
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames])
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h <= SKIN_H_LOW) | (h >= SKIN_H_HIGH))
            & (s >= SKIN_S_MIN) & (s <= SKIN_S_MAX) & (v >= SKIN_V_MIN))


def cube_visible(frames: np.ndarray) -> np.ndarray:
    """True for frames in the cube clip. Each video is a cube clip followed by a sponge clip;
    the cube is the only strongly green thing. The hand can hide the cube for a moment, so the
    clip is taken to end at the last frame with green pixels."""
    f = frames.astype(np.int16)
    green = ((f[..., 1] > f[..., 0] + 40) & (f[..., 1] > f[..., 2] + 40)).mean(axis=(1, 2))
    last_cube = np.flatnonzero(green > 0.005).max()
    return np.arange(len(frames)) <= last_cube


# %% Cut into grips
HAND_ON = 0.04        # hand present when skin pixels cover more than 4% of the frame
                      # (between grips the hand leaves the frame and this falls below 1%)
MIN_GRIP_S = 0.8      # shorter stretches are not grips (hand passing the edge of the frame)
MIN_GAP_S = 0.35      # shorter gaps are joined: a lift can take hand and object out of the top
                      # of the frame for up to 0.33 s; real gaps between grips are 0.4 s or more


def find_grips(hand_frac: np.ndarray, fps: int = FPS, hand_on: float = HAND_ON,
               min_grip_s: float = MIN_GRIP_S, min_gap_s: float = MIN_GAP_S) -> list:
    """Start and end frame (end excluded) of every stretch where the hand is in the frame.
    A stretch that touches the start or end of the video is kept if it is long enough
    (one take starts in the middle of a grip; its hold and release are recorded).
    The three settings can be changed to test how much the cutting depends on them."""
    on = hand_frac > hand_on
    starts = list(np.flatnonzero(on[1:] & ~on[:-1]) + 1)
    ends = list(np.flatnonzero(on[:-1] & ~on[1:]) + 1)
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        ends.append(len(on))
    runs = []
    for s, e in zip(starts, ends):                       # join runs split by a short gap
        if runs and s - runs[-1][1] < min_gap_s * fps:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    return [(s, e) for s, e in runs if e - s >= min_grip_s * fps]


# %% Grip features
# Object area, shape and lift are measured against the empty scene just before the grip (the
# median of the hand-free frames in the gap before it), so where the object rests in each take
# drops out. Hand area and motion come from the grip's own frames.
# Units: areas as fractions, positions in object heights, time in seconds.
def object_mask(frames: np.ndarray, obj: str) -> np.ndarray:
    """Cube: green pixels. Sponge: its yellow side (the dark scouring pad is too close in colour
    to the hand's shadow to separate)."""
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames]).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if obj == 'cube':
        return (h >= 40) & (h <= 90) & (s >= 90)
    return (h >= 21) & (h <= 38) & (s >= 90) & (v >= 90)


def empty_reference(frames: np.ndarray, hand_frac: np.ndarray, same_clip: np.ndarray,
                    start: int, end: int, video_grips) -> np.ndarray:
    """Median of the hand-free frames of the same clip in the gap just before the grip
    (start, end). If there are none (a take that starts mid-grip), the gap after it is used."""
    free = (hand_frac < HAND_ON / 2) & same_clip
    prev_end = max([e for e in video_grips.end if e <= start], default=0)
    idx = np.flatnonzero(free[prev_end:start]) + prev_end
    if len(idx) == 0:
        next_start = min([s for s in video_grips.start if s >= end], default=len(frames))
        idx = np.flatnonzero(free[end:next_start]) + end
    return np.median(frames[idx], axis=0).astype(np.uint8)


def largest_blob(mask: np.ndarray):
    """Largest connected region of a binary mask: (area, solidity, centroid y, height).
    Solidity = area / area of its convex hull; a bent or folded object has lower solidity."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return 0, np.nan, np.nan, np.nan
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[k, cv2.CC_STAT_AREA]
    contours, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(contours[0]))
    return area, area / hull if hull > 0 else np.nan, cent[k, 1], stats[k, cv2.CC_STAT_HEIGHT]


APPEAR_W, APPEAR_H = 32, 18     # size of the difference image used as "appearance" (attempt 7)


def grip_features(frames: np.ndarray, ref: np.ndarray, obj: str) -> tuple:
    """Measures of one grip (frames = the grip's frames, ref = empty scene before it).
    Returns (dict of numbers, appearance vector)."""
    hand = hand_mask(frames).mean(axis=(1, 2))
    om = object_mask(frames, obj)
    ref_area, ref_sol, ref_y, ref_h = largest_blob(object_mask(ref[None], obj)[0])
    blobs = np.array([largest_blob(m) for m in om], float)          # area, solidity, y, height
    visible = blobs[:, 0] / max(ref_area, 1)                         # 1 = as visible as at rest
    seen = visible > 0.3                                             # enough of it to measure shape
    lift = (ref_y - blobs[:, 2]) / max(ref_h, 1)                     # upward, in object heights
    lift = np.where(seen, lift, np.nan)
    gray = frames.astype(np.float32).mean(-1)
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))         # mean change frame to frame
    rise = np.diff(np.nan_to_num(lift, nan=0.0)) * FPS               # object heights per second
    sol = np.where(seen, blobs[:, 1], np.nan)
    f = dict(
        hand_mean=hand.mean(), hand_max=hand.max(), hand_std=hand.std(), duration_s=len(frames) / FPS,
        vis_min=visible.min(), vis_mean=visible.mean(), vis_std=visible.std(),
        lift_max=np.nanmax(lift) if seen.any() else 0.0, lift_mean=np.nanmean(lift) if seen.any() else 0.0,
        rise_max=rise.max(), motion_mean=motion.mean(), motion_max=motion.max(),
        sol_min=np.nanmin(sol) / ref_sol if seen.any() else 1.0,
        sol_mean=np.nanmean(sol) / ref_sol if seen.any() else 1.0,
    )
    # appearance: mean absolute difference from the empty scene over the grip, small and grey
    diff = np.abs(frames.astype(np.float32) - ref.astype(np.float32)).mean(axis=(0, 3))
    appearance = cv2.resize(diff, (APPEAR_W, APPEAR_H), interpolation=cv2.INTER_AREA).ravel()
    return f, appearance


def all_grip_features(grips_part, video_grips_all) -> tuple:
    """Features of every grip in grips_part (rows of grips.csv), in the same order.
    video_grips_all = grips.csv, used only for where each video's grips start and end
    (to find the gap before a grip); no labels are used. Returns (DataFrame, array)."""
    import pandas as pd
    feats, appear = {}, {}
    for video, vg in grips_part.groupby('video', sort=False):
        frames = read_frames(video)
        hand_frac = hand_mask(frames).mean(axis=(1, 2))
        cube = cube_visible(frames)
        spans = video_grips_all[video_grips_all.video == video]
        for row in vg.itertuples():
            same_clip = cube == cube[row.start]
            ref = empty_reference(frames, hand_frac, same_clip, row.start, row.end, spans)
            feats[row.grip_id], appear[row.grip_id] = grip_features(
                frames[row.start:row.end], ref, row.object)
    ids = list(grips_part.grip_id)
    return (pd.DataFrame([feats[g] for g in ids], index=ids),
            np.array([appear[g] for g in ids]))


# %% Hand-free frames
GRID_X, GRID_Y = 8, 6        # coarse grid for scene features: 8 x 6 cells, mean R, G, B of each
GAP_EDGE = 2                 # frames dropped at each end of a gap (hand still leaving/arriving)


def hand_free_frames(video: str, video_grips) -> list:
    """Rows (frame, object, gap number) for the frames of one video that are outside every grip
    and show no hand, dropping GAP_EDGE frames at both ends of each gap."""
    frames = read_frames(video)
    hand_frac = hand_mask(frames).mean(axis=(1, 2))
    cube = cube_visible(frames)
    in_grip = np.zeros(len(frames), bool)
    for g in video_grips.itertuples():
        in_grip[g.start:g.end] = True
    free = ~in_grip & (hand_frac < HAND_ON / 2)
    d = np.diff(np.r_[0, free.astype(int), 0])
    rows = []
    for gap, (s, e) in enumerate(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))):
        for i in range(s + GAP_EDGE, e - GAP_EDGE):
            rows.append(dict(frame=i, object='cube' if cube[i] else 'sponge', gap=gap))
    return rows


def grid_cells(h: int, w: int) -> list:
    """Row and column slices of the GRID_Y x GRID_X cells, row by row."""
    return [(slice(i * h // GRID_Y, (i + 1) * h // GRID_Y), slice(j * w // GRID_X, (j + 1) * w // GRID_X))
            for i in range(GRID_Y) for j in range(GRID_X)]


def scene_features(frame: np.ndarray) -> np.ndarray:
    """Mean R, G, B of every grid cell: 8 x 6 cells x 3 colours = 144 numbers."""
    h, w, _ = frame.shape
    return np.concatenate([frame[c].reshape(-1, 3).mean(axis=0) for c in grid_cells(h, w)])


def object_area(frames_free: np.ndarray) -> np.ndarray:
    """Pixels that are not plain table in the median hand-free frame (object and its shadow),
    grown by 12 pixels in every direction."""
    med = np.median(frames_free, axis=0)
    obj = ((med.max(-1) - med.min(-1)) > 40) | (med.max(-1) < 120)
    return cv2.dilate(obj.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0


def scene_table(grips_all, people: list) -> tuple:
    """Hand-free frames of the given people: (DataFrame of rows, scene features, mask of object
    area over all their clips)."""
    import pandas as pd
    rows, X, any_object = [], [], None
    for video, (person, label) in VIDEOS.items():
        if person not in people:
            continue
        frames = read_frames(video)
        part = pd.DataFrame(hand_free_frames(video, grips_all[grips_all.video == video]))
        part['person'], part['label'], part['video'] = person, label, video
        part['gap'] = video + '#' + part.gap.astype(str)
        for obj in ('cube', 'sponge'):
            sel = frames[part.frame[part.object == obj].values]
            area = object_area(sel)
            any_object = area if any_object is None else any_object | area
        X.append(np.stack([scene_features(f) for f in frames[part.frame.values]]))
        rows.append(part)
    return pd.concat(rows, ignore_index=True), np.vstack(X), any_object


def table_only(X: np.ndarray, any_object: np.ndarray) -> np.ndarray:
    """Keep only the grid cells that never contain the object."""
    h, w = any_object.shape
    keep = [k for k, c in enumerate(grid_cells(h, w)) if not any_object[c].any()]
    return X.reshape(len(X), -1, 3)[:, keep, :].reshape(len(X), -1)


def gap_accuracy(pred: np.ndarray, part) -> float:
    """Majority vote over the frames of each gap, then accuracy over gaps (frames of one gap are
    almost identical, so the gap is the unit, like the grip is for the grip methods)."""
    import pandas as pd
    votes = pd.DataFrame(dict(gap=part.gap.values, pred=pred, y=(part.label == 'firm').values))
    per_gap = votes.groupby('gap').agg(pred=('pred', 'mean'), y=('y', 'first'))
    return float(((per_gap.pred > 0.5) == per_gap.y).mean())
