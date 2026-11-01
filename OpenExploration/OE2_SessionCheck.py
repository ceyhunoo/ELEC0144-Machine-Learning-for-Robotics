# CODE MAP of OE2_SessionCheck.py  (can the takes be told apart with no hand in the picture?)
# Settings        which members are used (training members only), classifier
# Hand-free frames frames between grips with no hand in view, grouped by gap (oe_common)
# Scene features  mean colour of each cell of an 8 x 6 grid; second version keeps only the cells
#                 that never contain the object (table only)
# Check           logistic regression, firm take vs light take:
#                   across people (train on two members, test on the third, as in validation)
#                   within a person (train on half of each take's gaps, test on the other half)
# What differs    mean table colour and object resting position of each take
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Why: firm and light grips were filmed in separate takes. If the takes differ (light, camera,
# object position), a method could recognise the take instead of the grip. Frames with no hand
# in view carry no grip, so anything learned from them is the take.
# Run from the OpenExploration folder after OE1_Grips.py:  python OE2_SessionCheck.py
# Writes results/session_check.csv and results/take_differences.csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from oe_common import RESULTS_DIR, read_frames, scene_table, table_only, gap_accuracy

grips = pd.read_csv(RESULTS_DIR / 'grips.csv')
members = sorted(grips[grips.set == 'train'].person.unique())     # the test member is not used


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


# %% Hand-free frames
free, X_all, any_object = scene_table(grips, members)
y = (free.label == 'firm').values.astype(int)
print('Hand-free frames (gaps) per take:')
print(free.groupby(['person', 'label']).agg(frames=('frame', 'size'), gaps=('gap', 'nunique')).to_string(), '\n')


# %% Scene features
X_table = table_only(X_all, any_object)
print(f'Grid cells that never contain the object: {X_table.shape[1] // 3} of 48\n')


# %% Check
results = []
for name, X in [('whole frame', X_all), ('table only (object area left out)', X_table)]:
    accs = []
    for m in members:                                   # across people
        tr, te = (free.person != m).values, (free.person == m).values
        accs.append(gap_accuracy(model().fit(X[tr], y[tr]).predict(X[te]), free[te]))
    results.append(dict(features=name, test='across people (leave one member out)',
                        accuracy=round(np.mean(accs), 3),
                        per_member=', '.join(f'{m} {a:.0%}' for m, a in zip(members, accs))))
    accs = []
    for m in members:                                   # within a person, half the gaps each
        own = (free.person == m).values
        gap_no = free.gap.str.split('#').str[1].astype(int).values
        for k in (0, 1):
            tr, te = own & (gap_no % 2 == k), own & (gap_no % 2 != k)
            accs.append(gap_accuracy(model().fit(X[tr], y[tr]).predict(X[te]), free[te]))
    results.append(dict(features=name, test='within a person (half the gaps each)',
                        accuracy=round(np.mean(accs), 3), per_member='-'))
res = pd.DataFrame(results)
pd.set_option('display.width', 180)
print(res.to_string(index=False), '\n')
res.to_csv(RESULTS_DIR / 'session_check.csv', index=False)


# %% What differs
# Table colour: top-left 40 x 100 pixels (always plain table). Object position: centre of the
# non-table pixels in the median hand-free frame, in pixels of the 320 x 180 frame.
rows = []
for (video, person, label, obj), part in free.groupby(['video', 'person', 'label', 'object']):
    sel = read_frames(video)[part.frame.values].astype(float)
    med = np.median(sel, axis=0)
    obj_px = ((med.max(-1) - med.min(-1)) > 40) | (med.max(-1) < 120)
    ys, xs = np.nonzero(obj_px)
    r, g, b = sel[:, :40, :100].reshape(-1, 3).mean(axis=0)
    rows.append(dict(person=person, label=label, object=obj, table_R=r, table_G=g, table_B=b,
                     object_x=xs.mean(), object_y=ys.mean()))
diff = pd.DataFrame(rows).round(1)
print(diff.to_string(index=False))
diff.to_csv(RESULTS_DIR / 'take_differences.csv', index=False)
