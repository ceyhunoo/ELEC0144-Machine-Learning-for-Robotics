# CODE MAP of OE1_Grips.py  (run first; everything else reads its output)
# Settings        seed for the test-set draw, output file names
# Cut into grips  for every video: hand mask, cube or sponge clip, grip start and end frames
# Count table     grips per person, object and label (the report's data table)
# Test set        draw one member by lot; all of their grips become the test set (Rule 2)
# Robustness      the cutting settings were chosen looking at all eight videos (no labels);
#                 count grips per clip again for 45 other settings to show they do not matter
# Check figures   hand area over time with the grips shaded; one frame from every grip
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the OpenExploration folder:  python OE1_Grips.py
# Writes results/grips.csv, counts.csv, cutting_robustness.csv, fig_grip_cutting.png, fig_all_grips.png
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from oe_common import (VIDEOS, PEOPLE, FPS, RESULTS_DIR, read_frames, hand_mask,
                       cube_visible, find_grips, HAND_ON)
from itertools import product

TEST_DRAW_SEED = 2026        # fixed so the draw can be repeated; written down before exploring
RESULTS_DIR.mkdir(exist_ok=True)


# %% Cut into grips
rows = []
hand_curves = {}             # kept for the figure
for video, (person, label) in VIDEOS.items():
    frames = read_frames(video)
    hand_frac = hand_mask(frames).mean(axis=(1, 2))       # fraction of the frame that is skin
    cube = cube_visible(frames)
    hand_curves[video] = (hand_frac, cube)
    for k, (s, e) in enumerate(find_grips(hand_frac)):
        obj = 'cube' if cube[s:e].mean() > 0.5 else 'sponge'
        rows.append(dict(video=video, person=person, label=label, object=obj,
                         start=int(s), end=int(e),
                         start_s=round(s / FPS, 2), end_s=round(e / FPS, 2)))

grips = pd.DataFrame(rows)
grips['n'] = grips.groupby(['video', 'object']).cumcount() + 1        # 1..10 within a clip
grips['grip_id'] = (grips.person + '-' + grips.label + '-' + grips.object + '-'
                    + grips.n.astype(str).str.zfill(2))


# %% Count table
counts = grips.pivot_table(index='person', columns=['object', 'label'], values='grip_id',
                           aggfunc='count', fill_value=0)
counts['total'] = counts.sum(axis=1)
counts.loc['total'] = counts.sum()
print(counts, '\n')
counts.to_csv(RESULTS_DIR / 'counts.csv')


# %% Test set
# Rule 2: about 25% aside before exploring, never used for training. With four members,
# all grips of one member is exactly 25%, and every test grip comes from takes the method has
# never seen (useful because each take holds only one label).
rng = np.random.default_rng(TEST_DRAW_SEED)
test_person = str(rng.choice(sorted(PEOPLE)))
grips['set'] = np.where(grips.person == test_person, 'test', 'train')
print(f'Test member (drawn with seed {TEST_DRAW_SEED}): {test_person}')
print(grips.groupby('set').size().to_string(), '\n')
grips.to_csv(RESULTS_DIR / 'grips.csv', index=False)


# %% Robustness
# For every combination of hand threshold, shortest grip and longest joined gap, count the grips
# of each clip. If the test member's clips still give 10 grips each, the choice of settings
# (made looking at all videos) cannot have shaped the test set.
settings = list(product([0.02, 0.03, 0.04, 0.06, 0.08],      # hand threshold
                        [0.5, 0.8, 1.2],                      # shortest grip (s)
                        [0.2, 0.35, 0.6]))                    # gaps shorter than this are joined (s)
rob = []
for video, (person, label) in VIDEOS.items():
    hand_frac, cube = hand_curves[video]
    for hand_on, min_grip, min_gap in settings:
        spans = find_grips(hand_frac, hand_on=hand_on, min_grip_s=min_grip, min_gap_s=min_gap)
        n_cube = sum(cube[s:e].mean() > 0.5 for s, e in spans)
        rob.append(dict(person=person, label=label, hand_on=hand_on, min_grip_s=min_grip,
                        min_gap_s=min_gap, cube=n_cube, sponge=len(spans) - n_cube))
rob = pd.DataFrame(rob)
rob['all_10'] = (rob.cube == 10) & (rob.sponge == 10)
both = rob.groupby(['person', 'hand_on', 'min_grip_s', 'min_gap_s']).all_10.all().groupby('person').sum()
print(f'Settings (of {len(settings)}) giving exactly 10 grips in every clip of a member\'s two takes:')
print(both.to_string(), '\n')
rob.to_csv(RESULTS_DIR / 'cutting_robustness.csv', index=False)


# %% Check figures
# 1) hand area over time with every detected grip shaded (all eight videos)
fig, axs = plt.subplots(len(VIDEOS), 1, figsize=(12, 13), sharey=True)
for ax, video in zip(axs, VIDEOS):
    hand_frac, cube = hand_curves[video]
    t = np.arange(len(hand_frac)) / FPS
    ax.plot(t, hand_frac, 'k', lw=0.8)
    ax.axhline(HAND_ON, color='grey', ls=':', lw=0.8)
    ax.axvline(cube.sum() / FPS, color='C2', lw=1.5)                   # cube clip -> sponge clip
    for g in grips[grips.video == video].itertuples():
        ax.axvspan(g.start_s, g.end_s, color='C0' if g.object == 'cube' else 'C1', alpha=0.25)
    person, label = VIDEOS[video]
    ax.set_title(f'{person}, {label}  ({video})', fontsize=9, loc='left')
    ax.set_ylabel('hand area')
axs[-1].set_xlabel('time (s)')
fig.suptitle('Grip cutting: skin-coloured fraction of the frame; shaded = one grip '
             '(blue cube, orange sponge); green line = cube clip ends; dotted = threshold', fontsize=10)
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'fig_grip_cutting.png', dpi=110)

# 2) the middle frame of every grip, one row per clip, to check each one by eye
clips = grips.groupby(['video', 'object'], sort=False)
fig, axs = plt.subplots(len(clips), 10, figsize=(15, 1.0 * len(clips)))
for r, ((video, obj), g) in enumerate(clips):
    frames = read_frames(video)
    for c in range(10):
        ax = axs[r, c]
        ax.axis('off')
        if c < len(g):
            row = g.iloc[c]
            ax.imshow(frames[(row.start + row.end) // 2])
            if c == 0:
                ax.set_title(f'{row.person} {row.label} {obj}', fontsize=7, loc='left')
fig.tight_layout(pad=0.2)
fig.savefig(RESULTS_DIR / 'fig_all_grips.png', dpi=90)
print('Figures saved in', RESULTS_DIR)
