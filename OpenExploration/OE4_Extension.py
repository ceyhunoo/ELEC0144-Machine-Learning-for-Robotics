# CODE MAP of OE4_Extension.py  (Going further B: a deeper test by ablation)
# Settings        the measure groups and the ablations (what each one removes)
# Data            training grips (fit) and test grips (the test member's, evaluated only, never fitted)
# Ablations       each version trained on the 120 training grips, run once on the 40 test grips
# Scene only      hand and object both removed: hand-free frames between grips, trained on the
#                 training members' takes, tested on the test member's takes
# Why the cube    measures per person and label; cube models with one group of measures each;
#                 weights of the final cube model; how far the test member's hand area is off
# Figure          test accuracy of each version; cube hand area and lift, sponge area per person
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the OpenExploration folder after OE1_Grips.py:  python OE4_Extension.py
# Writes results/extension_ablations.csv, per_person_cues.csv, why_cube.csv, cube_model_weights.csv,
# fig_extension.png
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from oe_common import RESULTS_DIR, all_grip_features, scene_table, table_only, gap_accuracy

HAND = ['hand_mean', 'hand_max', 'hand_std']                  # from the skin mask
OBJECT = ['vis_min', 'vis_mean', 'vis_std', 'sol_min', 'sol_mean']   # object area and shape
LIFT = ['lift_max', 'lift_mean', 'rise_max']                  # object movement (object mask)
MOTION = ['motion_mean', 'motion_max']                        # whole-scene change frame to frame

# name: (measures of the cube model, measures of the sponge model)
ABLATIONS = {
    'Final method':                    (HAND + OBJECT + LIFT + MOTION, HAND + OBJECT),
    'Without movement':                (HAND + OBJECT,                 HAND + OBJECT),
    'Hand hidden (no hand, no motion)': (OBJECT + LIFT,                 OBJECT),
    'Object hidden (no object, no lift)': (HAND + MOTION,               HAND),
}


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


# %% Data
grips = pd.read_csv(RESULTS_DIR / 'grips.csv')
train = grips[grips.set == 'train'].reset_index(drop=True)
test = grips[grips.set == 'test'].reset_index(drop=True)
test_person = test.person.iloc[0]
f_train, _ = all_grip_features(train, grips)
f_test, _ = all_grip_features(test, grips)
y_train = (train.label == 'firm').values.astype(int)
y_test = (test.label == 'firm').values.astype(int)


# %% Ablations
rows = []
for name, (cube_cols, sponge_cols) in ABLATIONS.items():
    correct, called_firm = np.zeros(len(test), bool), {}
    for obj, cols in (('cube', cube_cols), ('sponge', sponge_cols)):
        tr, te = (train.object == obj).values, (test.object == obj).values
        model = logistic().fit(f_train.loc[train.grip_id[tr], cols].values, y_train[tr])
        pred = model.predict(f_test.loc[test.grip_id[te], cols].values)
        correct[te] = pred == y_test[te]
        called_firm[obj] = int(pred.sum())
    rows.append(dict(version=name, test=correct.mean(),
                     cube=correct[(test.object == 'cube').values].mean(),
                     sponge=correct[(test.object == 'sponge').values].mean(), unit='40 grips',
                     cube_called_firm=called_firm['cube'], sponge_called_firm=called_firm['sponge']))


# %% Scene only
# Hand and object both removed: frames between grips with no hand in view, table cells only
# (cells chosen from the training members' frames). If this beats guessing on the test member,
# the takes themselves give the label away.
free_tr, X_tr, obj_tr = scene_table(grips, sorted(train.person.unique()))
free_te, X_te, _ = scene_table(grips, [test_person])
model = logistic().fit(table_only(X_tr, obj_tr), (free_tr.label == 'firm').values.astype(int))
pred = model.predict(table_only(X_te, obj_tr))
acc = {o: gap_accuracy(pred[(free_te.object == o).values], free_te[free_te.object == o])
       for o in ('cube', 'sponge')}
rows.append(dict(version='Hand and object hidden (scene only)', test=gap_accuracy(pred, free_te),
                 cube=acc['cube'], sponge=acc['sponge'], unit=f'{free_te.gap.nunique()} gaps'))

abl = pd.DataFrame(rows)
pd.set_option('display.width', 160)
print(f'All versions trained on the training members, tested on {test_person}:\n')
print(abl.round(3).to_string(index=False), '\n')
abl.to_csv(RESULTS_DIR / 'extension_ablations.csv', index=False)


# %% Why the cube
# Mean of three measures per person and label, all four members (the test member's numbers are
# only looked at here, never used to fit anything).
both = pd.concat([f_train.assign(person=train.person.values, label=train.label.values, object=train.object.values),
                  f_test.assign(person=test.person.values, label=test.label.values, object=test.object.values)])
cues = (both.groupby(['object', 'person', 'label'])[['lift_max', 'vis_min', 'hand_max']]
        .mean().round(3).unstack('label'))
print(cues.to_string(), '\n')
cues.to_csv(RESULTS_DIR / 'per_person_cues.csv')
sp = both[both.object == 'sponge']
print('Sponge, smallest visible yellow area (fraction of resting area), all 80 grips: '
      f'firm at most {sp[sp.label == "firm"].vis_min.max():.2f}, '
      f'light at least {sp[sp.label == "light"].vis_min.min():.2f}\n')

# Cube models with one group of measures each: how many of the 20 test cube grips are called firm.
tr, te = (train.object == 'cube').values, (test.object == 'cube').values
why = []
for name, cols in [('final cube model', HAND + OBJECT + LIFT + MOTION), ('hand area only', HAND),
                   ('object area and shape only', OBJECT), ('lift only', LIFT), ('motion only', MOTION)]:
    model = logistic().fit(f_train.loc[train.grip_id[tr], cols].values, y_train[tr])
    pred = model.predict(f_test.loc[test.grip_id[te], cols].values)
    why.append(dict(cube_model=name, accuracy=(pred == y_test[te]).mean(), called_firm=int(pred.sum())))
why = pd.DataFrame(why)
print(f'Cube models on the {te.sum()} test cube grips:')
print(why.to_string(index=False), '\n')
final_cube = logistic().fit(f_train.loc[train.grip_id[tr], ABLATIONS['Final method'][0]].values, y_train[tr])
coef = pd.Series(final_cube[-1].coef_[0], index=ABLATIONS['Final method'][0]).round(2)
print('Final cube model, weight of each standardised measure:\n', coef.to_string(), '\n')
z = ((f_test.loc[test.grip_id[te], HAND].mean() - f_train.loc[train.grip_id[tr], HAND].mean())
     / f_train.loc[train.grip_id[tr], HAND].std()).round(2)
print(f'{test_person}\'s cube hand area, in training standard deviations from the training mean:\n', z.to_string(), '\n')
why.to_csv(RESULTS_DIR / 'why_cube.csv', index=False)
pd.concat([coef.rename('weight'), z.rename('test_member_sd_from_train_mean')], axis=1).to_csv(
    RESULTS_DIR / 'cube_model_weights.csv')


# %% Figure
fig, axs = plt.subplots(2, 2, figsize=(9, 6.2))
ax = axs[0, 0]
x = np.arange(len(abl))
ax.axhspan(0.35, 0.65, color='grey', alpha=0.15, label='guessing, 95% range (40 grips)')
ax.bar(x - 0.2, abl.cube, 0.4, label='cube', color='C2')
ax.bar(x + 0.2, abl.sponge, 0.4, label='sponge', color='C1')
ax.set_xticks(x, ['Final', 'No\nmovement', 'Hand\nhidden', 'Object\nhidden', 'Scene\nonly'], fontsize=8)
ax.set_ylim(0, 1.3); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel(f'test accuracy ({test_person})', fontsize=9)
ax.set_title('(a) Ablations on the test member', fontsize=9)
ax.legend(fontsize=7, loc='upper center', ncol=3, handlelength=1.2, columnspacing=0.8)
people = sorted(train.person.unique()) + [test_person]
panels = [(axs[0, 1], 'cube', 'hand_max', 'largest hand area\n(fraction of the frame)', '(b) Cube: hand area'),
          (axs[1, 0], 'cube', 'lift_max', 'highest lift (object heights)', '(c) Cube: lift'),
          (axs[1, 1], 'sponge', 'vis_min', 'smallest visible yellow area\n(fraction of resting area)', '(d) Sponge: visible yellow area')]
for ax, obj, col, ylab, title in panels:
    sub = both[both.object == obj]
    for j, p in enumerate(people):
        for lab, dx, c in (('light', -0.15, 'C0'), ('firm', 0.15, 'C3')):
            v = sub[(sub.person == p) & (sub.label == lab)][col]
            ax.scatter(np.full(len(v), j + dx), v, s=8, color=c, alpha=0.6, label=lab if j == 0 else None)
    ax.set_xticks(range(len(people)), [p + ('\n(test)' if p == test_person else '') for p in people], fontsize=8)
    ax.set_ylabel(ylab, fontsize=8); ax.set_title(title + ', one dot per grip', fontsize=9)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'fig_extension.png', dpi=150)
print('Figure saved:', RESULTS_DIR / 'fig_extension.png')
