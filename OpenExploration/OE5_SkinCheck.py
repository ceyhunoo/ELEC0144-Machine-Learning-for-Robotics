# CODE MAP of OE5_SkinCheck.py  (do the skin-colour bounds shape the test result?)
# Settings        the skin-colour bounds as chosen in oe_common.py, the step each bound is moved by,
#                 and the settings tried: each bound moved down and up on its own, all bounds
#                 widened together, all narrowed together
# Run one setting with that hand mask: cut every video into grips (as OE1_Grips.py), measure every
#                 grip, train the final method (as FinalMethod.py) on the training members, run it
#                 once on the test member
# Run all         every setting: grips per clip, training and test grips, test accuracy
# Save            results/skin_check.csv
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the OpenExploration folder after OE1_Grips.py:  python OE5_SkinCheck.py  (about 10 minutes)
# Why: the skin-colour bounds in oe_common.py were set looking at all eight videos (no labels),
# test videos included. If the test member's grips and the test result stay the same when every
# bound moves, the choice of bounds cannot have shaped the test result. Nothing here is used to
# choose anything: the final method, its measures and the test member are those of FinalMethod.py
# and OE1_Grips.py. The steps and the settings below were fixed before this file was run.
# A move is left out if it would let the sponge's or the cube's colour pass as skin (their hue
# ranges in oe_common.object_mask: sponge H 21-38, cube H 40-90).
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import oe_common
from oe_common import (VIDEOS, RESULTS_DIR, read_frames, hand_mask, cube_visible, find_grips,
                       all_grip_features)

CHOSEN = dict(SKIN_H_LOW=20, SKIN_H_HIGH=165, SKIN_S_MIN=35, SKIN_S_MAX=200, SKIN_V_MIN=50)
STEP = dict(SKIN_H_LOW=3, SKIN_H_HIGH=5, SKIN_S_MIN=15, SKIN_S_MAX=15, SKIN_V_MIN=15)
WIDER = dict(SKIN_H_LOW=+1, SKIN_H_HIGH=-1, SKIN_S_MIN=-1, SKIN_S_MAX=+1, SKIN_V_MIN=-1)  # sign that lets more pixels in
SPONGE_H_MIN = 21                                       # oe_common.object_mask: sponge yellow starts at H 21
assert all(getattr(oe_common, k) == v for k, v in CHOSEN.items())     # the bounds actually in use

HAND = ['hand_mean', 'hand_max', 'hand_std']
OBJECT = ['vis_min', 'vis_mean', 'vis_std', 'sol_min', 'sol_mean']
MOVE = ['lift_max', 'lift_mean', 'rise_max', 'motion_mean', 'motion_max']
MEASURES = {'cube': HAND + OBJECT + MOVE, 'sponge': HAND + OBJECT}   # as in FinalMethod.py

test_person = pd.read_csv(RESULTS_DIR / 'grips.csv').query("set == 'test'").person.iloc[0]   # drawn in OE1


def allowed(bounds: dict) -> bool:
    """False if skin would take in the sponge's yellow hue (H 21 and up)."""
    return bounds['SKIN_H_LOW'] < SPONGE_H_MIN


settings = [('chosen', dict(CHOSEN))]
for k in CHOSEN:
    for sign, word in ((-1, 'down'), (+1, 'up')):
        b = dict(CHOSEN); b[k] = CHOSEN[k] + sign * STEP[k]
        settings.append((f'{k} {word} ({b[k]})', b))
settings.append(('all wider', {k: CHOSEN[k] + WIDER[k] * STEP[k] for k in CHOSEN}))
settings.append(('all narrower', {k: CHOSEN[k] - WIDER[k] * STEP[k] for k in CHOSEN}))
left_out = [name for name, b in settings if not allowed(b)]
settings = [(name, b) for name, b in settings if allowed(b)]
# 'all wider' would raise H_LOW into the sponge's hue; it is kept with H_LOW at its chosen value
if 'all wider' in left_out:
    b = {k: CHOSEN[k] + WIDER[k] * STEP[k] for k in CHOSEN}; b['SKIN_H_LOW'] = CHOSEN['SKIN_H_LOW']
    settings.append(('all wider (H_LOW kept)', b)); left_out.remove('all wider')


# %% Run one setting
def run_setting(bounds: dict) -> dict:
    """Cut, measure, train and test with the given skin bounds. Returns counts and accuracies."""
    for k, v in bounds.items():
        setattr(oe_common, k, v)                        # hand_mask reads these at every call
    rows = []
    for video, (person, label) in VIDEOS.items():       # cut into grips, as OE1_Grips.py
        frames = read_frames(video)
        hand_frac = hand_mask(frames).mean(axis=(1, 2))
        cube = cube_visible(frames)
        for s, e in find_grips(hand_frac):
            obj = 'cube' if cube[s:e].mean() > 0.5 else 'sponge'
            rows.append(dict(video=video, person=person, label=label, object=obj, start=int(s), end=int(e)))
    grips = pd.DataFrame(rows)
    grips['n'] = grips.groupby(['video', 'object']).cumcount() + 1
    grips['grip_id'] = grips.person + '-' + grips.label + '-' + grips.object + '-' + grips.n.astype(str)
    grips['set'] = np.where(grips.person == test_person, 'test', 'train')
    per_clip = grips.groupby(['video', 'object']).size()
    train = grips[grips.set == 'train'].reset_index(drop=True)
    test = grips[grips.set == 'test'].reset_index(drop=True)

    f_train, _ = all_grip_features(train, grips)       # measure, as FinalMethod.py
    f_test, _ = all_grip_features(test, grips)
    y_train = (train.label == 'firm').values.astype(int)
    y_test = (test.label == 'firm').values.astype(int)
    pred = np.zeros(len(test), int)
    for obj, cols in MEASURES.items():                  # train on training members, test once
        tr, te = (train.object == obj).values, (test.object == obj).values
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        model.fit(f_train.loc[train.grip_id[tr], cols].values, y_train[tr])
        pred[te] = model.predict(f_test.loc[test.grip_id[te], cols].values)
    correct = pred == y_test
    is_test = np.array([VIDEOS[v][0] == test_person for v in per_clip.index.get_level_values(0)])
    test_clips = per_clip[is_test]                      # the test member's four clips
    return dict(train_grips=len(train), test_grips=len(test),
                clips_with_10=int((per_clip == 10).sum()), clips=len(per_clip),
                test_clips_all_10=bool((test_clips == 10).all() and len(test_clips) == 4),
                test_correct=int(correct.sum()), test_accuracy=correct.mean(),
                cube=correct[(test.object == 'cube').values].mean(),
                sponge=correct[(test.object == 'sponge').values].mean())


# %% Run all
out = []
for name, bounds in settings:
    r = run_setting(bounds)
    out.append(dict(setting=name, **{k.replace('SKIN_', ''): v for k, v in bounds.items()}, **r))
    print(f'{name:24s} test {r["test_correct"]}/{r["test_grips"]} = {r["test_accuracy"]:.1%} '
          f'(cube {r["cube"]:.0%}, sponge {r["sponge"]:.0%}); clips with 10 grips: '
          f'{r["clips_with_10"]}/{r["clips"]}; test clips all 10: {r["test_clips_all_10"]}')
for k, v in CHOSEN.items():
    setattr(oe_common, k, v)                            # back to the chosen bounds
if left_out:
    print('Left out (skin would take in the sponge\'s hue):', ', '.join(left_out))


# %% Save
res = pd.DataFrame(out)
res.to_csv(RESULTS_DIR / 'skin_check.csv', index=False)
print(f'\n{len(res)} settings: test accuracy {res.test_accuracy.min():.1%} to {res.test_accuracy.max():.1%}; '
      f'test member\'s clips give 10 grips each in {res.test_clips_all_10.sum()} of {len(res)}')
print('Saved:', RESULTS_DIR / 'skin_check.csv')
