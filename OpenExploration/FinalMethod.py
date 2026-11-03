# CODE MAP of FinalMethod.py  (runs the final method, attempt 8 of the tree, on the test set)
# Settings        the measures used by the cube model and by the sponge model
# Data            grips.csv from OE1_Grips.py: which grips are training and which are test
# Measures        hand area, object area/shape and movement of every grip (oe_common.grip_features)
# Train           one logistic regression per object, fitted on the training grips only
# Test            predict every test grip; accuracy overall, per object, per label
# Compare         with guessing: 50%, the range guessing reaches 95% of the time, and a p-value
# Figure          input and output of the method for four test grips
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the OpenExploration folder after OE1_Grips.py:  python FinalMethod.py
# Test set: all grips of the member drawn in OE1_Grips.py (Yikhui firm.mp4, Yikhui gentle.mp4).
from math import comb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from oe_common import RESULTS_DIR, FPS, all_grip_features, read_frames, hand_mask, object_mask

HAND = ['hand_mean', 'hand_max', 'hand_std']                               # skin area in the frame
OBJECT = ['vis_min', 'vis_mean', 'vis_std', 'sol_min', 'sol_mean']         # object area and shape
MOVE = ['lift_max', 'lift_mean', 'rise_max', 'motion_mean', 'motion_max']  # lift and scene motion
MEASURES = {'cube': HAND + OBJECT + MOVE,     # the cube does not deform: movement is needed
            'sponge': HAND + OBJECT}          # the sponge deforms: movement only added noise


# %% Data
grips = pd.read_csv(RESULTS_DIR / 'grips.csv')
train = grips[grips.set == 'train'].reset_index(drop=True)
test = grips[grips.set == 'test'].reset_index(drop=True)
assert not set(train.grip_id) & set(test.grip_id)                 # no grip in both sets
assert not set(train.video) & set(test.video)                     # no take in both sets
print('Test recordings:', ', '.join(sorted(test.video.unique())), f'({len(test)} grips)')
print('Training recordings:', ', '.join(sorted(train.video.unique())), f'({len(train)} grips)\n')


# %% Measures
f_train, _ = all_grip_features(train, grips)
f_test, _ = all_grip_features(test, grips)
y_train = (train.label == 'firm').values.astype(int)
y_test = (test.label == 'firm').values.astype(int)


# %% Train
# StandardScaler learns each measure's mean and spread from the training grips only;
# the test grips are scaled with those numbers.
models = {}
for obj, cols in MEASURES.items():
    rows = (train.object == obj).values
    models[obj] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    models[obj].fit(f_train.loc[train.grip_id[rows], cols].values, y_train[rows])


# %% Test
firm_score = np.zeros(len(test))
for obj, cols in MEASURES.items():
    rows = (test.object == obj).values
    firm_score[rows] = models[obj].predict_proba(f_test.loc[test.grip_id[rows], cols].values)[:, 1]
pred = (firm_score > 0.5).astype(int)
correct = pred == y_test
n, k = len(test), int(correct.sum())

print(f'Final method on the test set: {k}/{n} correct = {k / n:.1%}')
for obj in ('cube', 'sponge'):
    r = (test.object == obj).values
    print(f'  {obj:6s}: {correct[r].sum()}/{r.sum()} = {correct[r].mean():.1%}')
table = pd.crosstab(pd.Series(np.where(y_test, 'firm', 'light'), name='true'),
                    pd.Series(np.where(pred, 'firm', 'light'), name='predicted'))
print('\n', table, '\n')


# %% Compare
# Guessing (coin flip per grip) gets X ~ Binomial(n, 0.5) correct.
def p_at_least(k: int, n: int) -> float:
    """P(guessing gets k or more of n right)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


cdf = np.cumsum([comb(n, i) / 2 ** n for i in range(n + 1)])
lo, hi = int(np.searchsorted(cdf, 0.025)), int(np.searchsorted(cdf, 0.975))
print(f'Guessing: 50% on average (the test set has {y_test.sum()} firm and {n - y_test.sum()} light grips).')
print(f'Guessing gets between {lo}/{n} ({lo / n:.0%}) and {hi}/{n} ({hi / n:.0%}) right 95% of the time.')
print(f'Chance that guessing gets {k}/{n} or better: p = {p_at_least(k, n):.2g}')

out = test[['grip_id', 'video', 'object', 'label', 'start_s', 'end_s']].copy()
out['firm_score'] = firm_score.round(3)
out['predicted'] = np.where(pred, 'firm', 'light')
out['correct'] = correct
out.to_csv(RESULTS_DIR / 'final_test_predictions.csv', index=False)
pd.DataFrame([dict(n=n, correct=k, accuracy=k / n,
                   cube=correct[(test.object == 'cube').values].mean(),
                   sponge=correct[(test.object == 'sponge').values].mean(),
                   guess_low=lo / n, guess_high=hi / n, p_value=p_at_least(k, n))]
             ).to_csv(RESULTS_DIR / 'final_test_result.csv', index=False)
wrong = out[~out.correct]
print('\nWrong grips:\n', wrong[['grip_id', 'firm_score', 'start_s']].to_string(index=False))


# %% Figure
# For one firm and one light grip of each object: five frames (input), two of the measured
# curves, and the firm score (output). Sized for the report (about 17 cm wide), text about 6-7 pt.
plt.rcParams.update({'font.size': 6.5})
examples = [test[(test.object == o) & (test.label == l)].iloc[4] for o in ('sponge', 'cube') for l in ('firm', 'light')]
fig, axs = plt.subplots(4, 7, figsize=(7.2, 3.9),
                        gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 1.6, 0.65]))
for r, row in enumerate(examples):
    frames = read_frames(row.video)[row.start:row.end]
    for c, i in enumerate(np.linspace(0, len(frames) - 1, 5).astype(int)):
        axs[r, c].imshow(frames[i][15:165, 90:320])
        axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        if r == 0:
            axs[r, c].set_title(f'{i / FPS:.1f} s', fontsize=6.5)
    axs[r, 0].set_ylabel(f'{row.object}\n{row.label}', fontsize=6.5)
    t = np.arange(len(frames)) / FPS
    om = object_mask(frames, row.object).mean(axis=(1, 2))
    axs[r, 5].plot(t, hand_mask(frames).mean(axis=(1, 2)), lw=0.9, label='hand (fraction of frame)')
    axs[r, 5].plot(t, om / om[:3].mean() * 0.1, lw=0.9, label='object, scaled (0.1 = grip start)')
    axs[r, 5].set_ylim(0, 0.25); axs[r, 5].tick_params(labelsize=5.5)
    axs[r, 5].set_ylabel('area', fontsize=6)
    if r == 0:
        axs[r, 5].set_title('measured', fontsize=6.5)
    if r == 3:
        axs[r, 5].set_xlabel('time in grip (s)', fontsize=6)
    s = firm_score[test.index[test.grip_id == row.grip_id][0]]
    axs[r, 6].bar([0], [s], color='C3' if s > 0.5 else 'C0'); axs[r, 6].axhline(0.5, color='k', lw=0.6, ls=':')
    axs[r, 6].set_ylim(0, 1); axs[r, 6].set_xticks([]); axs[r, 6].tick_params(labelsize=5.5)
    axs[r, 6].set_ylabel('firm score', fontsize=6)
    axs[r, 6].text(0, min(s + 0.05, 0.85), f'{s:.2f}', ha='center', fontsize=6)
    if r == 0:
        axs[r, 6].set_title('output', fontsize=6.5)
fig.suptitle(f'Final method on four test grips ({test.person.iloc[0]}): input frames, measured curves, output', fontsize=7)
handles, labels = axs[0, 5].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=2, fontsize=6, frameon=False, bbox_to_anchor=(0.01, 0.0))
fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3, rect=(0, 0.05, 1, 1))
fig.savefig(RESULTS_DIR / 'fig_final_method_io.png', dpi=300)
print('\nFigure saved:', RESULTS_DIR / 'fig_final_method_io.png')
