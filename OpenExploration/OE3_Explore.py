# CODE MAP of OE3_Explore.py  (the attempts in the tree; training members only)
# Settings        classifier, validation scheme, which grips are used (set == 'train' only)
# Features        measures of every training grip (oe_common.grip_features), cached in cache/
# Validation      leave one member out: train on two members, validate on the third, repeat;
#                 accuracy over all 120 validation predictions, also per object and per member
# Attempts        one section per node of the tree, in the order they were tried
# Save            results/tree_results.csv (every number in the tree comes from here)
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the OpenExploration folder after OE1_Grips.py:  python OE3_Explore.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from oe_common import RESULTS_DIR, CACHE_DIR, all_grip_features

grips = pd.read_csv(RESULTS_DIR / 'grips.csv')
train = grips[grips.set == 'train'].reset_index(drop=True)      # the test member never appears here
assert 'test' not in set(train.set)
y = (train.label == 'firm').values.astype(int)
members = sorted(train.person.unique())


def logistic():
    """Standardise each feature (mean and spread from the training part only), then
    logistic regression. C = 1 is the default amount of regularisation."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


# %% Features
cache = CACHE_DIR / 'features_train.csv'
feats = pd.read_csv(cache, index_col=0) if cache.exists() else None
if feats is not None and list(feats.index) == list(train.grip_id):   # cache matches these grips
    appear = np.load(CACHE_DIR / 'appearance_train.npy')
else:                                                                  # missing or out of date: rebuild
    feats, appear = all_grip_features(train, grips)
    feats.to_csv(cache)
    np.save(CACHE_DIR / 'appearance_train.npy', appear)
assert list(feats.index) == list(train.grip_id)


# %% Validation
def validate(X: np.ndarray, make_model=logistic, per_object: bool = False, X_sponge=None) -> dict:
    """Leave one member out. Returns validation accuracy overall, per object, per member.
    per_object=True trains a separate model for cube grips and for sponge grips.
    X_sponge (optional): different features for the sponge model (X is then used for the cube)."""
    pred = np.zeros(len(train), int)
    for m in members:
        val = (train.person == m).values
        groups = [train.object == o for o in ('cube', 'sponge')] if per_object else [np.ones(len(train), bool)]
        for k, g in enumerate(groups):
            g = np.asarray(g)
            Xg = X_sponge if (per_object and k == 1 and X_sponge is not None) else X
            model = make_model().fit(Xg[~val & g], y[~val & g])
            pred[val & g] = model.predict(Xg[val & g])
    ok = pred == y
    out = dict(val=ok.mean(),
               val_cube=ok[train.object == 'cube'].mean(), val_sponge=ok[train.object == 'sponge'].mean())
    out.update({f'val_{m}': ok[train.person == m].mean() for m in members})
    return out


results = []


def record(node: str, why: str, cols, X, **kw):
    r = validate(X, **kw)
    results.append(dict(node=node, why=why, features=' '.join(cols), **r))
    print(f'{node:28s} val {r["val"]:.1%}  (cube {r["val_cube"]:.1%}, sponge {r["val_sponge"]:.1%}; '
          + ', '.join(f'{m} {r["val_" + m]:.0%}' for m in members) + ')')


HAND = ['hand_mean', 'hand_max', 'hand_std']
OBJECT = ['vis_min', 'vis_mean', 'vis_std', 'sol_min', 'sol_mean']

# %% Attempt 1: hand area
# Why: a firm grip may wrap more of the hand round the object, so more skin is in view.
record('1 Hand area', 'firm grip may wrap more hand round the object', HAND, feats[HAND].values)

# %% Attempt 2: object visibility and shape
# Why: a firm squeeze should deform the sponge (bend it, hide its yellow side) and cover more of
# the cube; this is a direct physical effect of force rather than a habit of the person.
record('2 Object area and shape', 'firm squeeze deforms the sponge, covers more of the cube',
       OBJECT, feats[OBJECT].values)

# %% Attempt 3: one model per object
# Why (from 2): the same measure means different things for the two objects (the sponge's yellow
# side halves when folded, the cube only gets covered), so one shared boundary fits neither.
record('3 Object, model per object', 'from 2: same measure means different things per object',
       OBJECT, feats[OBJECT].values, per_object=True)

# %% Attempt 4: hand area added to the per-object model
# Why (from 1 and 3): the cube does not deform, so object measures are at guessing on it (3);
# hand area was the only measure above guessing on the cube (1).
HO = HAND + OBJECT
record('4 + hand area', 'from 1, 3: cube needs a hand cue', HO, feats[HO].values, per_object=True)

# %% Attempt 5: movement added
# Why (from 4): the cube is still weak; holding something firmly may change how it is lifted
# (height, speed) and how much the scene moves.
MOVE = ['lift_max', 'lift_mean', 'rise_max', 'motion_mean', 'motion_max']
HOM = HO + MOVE
record('5 + movement', 'from 4: cube still weak; firm hold may lift differently', HOM,
       feats[HOM].values, per_object=True)

# %% Attempt 6: grip duration added
# Why (from 5): firm grips in the training members last longer on average; squeezing may take
# time. Risk: with firm and light filmed in separate takes, duration may be the pace of a take
# rather than the force, so it is only kept if it clearly helps.
HOMD = HOM + ['duration_s']
record('6 + grip duration', 'from 5: firm grips last longer; may be take pace', HOMD,
       feats[HOMD].values, per_object=True)

# %% Attempt 7: appearance instead of chosen measures
# Why (from 3-5): every measure so far was picked by us; a difference image (how each part of the
# scene differs from the empty table, averaged over the grip, 32 x 18 pixels) lets the model find
# its own cues. PCA keeps 10 directions so 40 grips per object are enough to fit.
def pca_logistic():
    return make_pipeline(StandardScaler(), PCA(n_components=10, random_state=0),
                         LogisticRegression(C=1.0, max_iter=5000))


record('7 Appearance (PCA)', 'from 3-5: let the model find its own cues', ['appearance_32x18'],
       appear, make_model=pca_logistic, per_object=True)

# %% Attempt 8: different measures for each object
# Why (from 4 and 5): movement raised the cube from 60% to 77% but lowered the sponge from 97% to
# 92%; so the cube model gets hand + object + movement and the sponge model hand + object only.
record('8 Own measures per object', 'from 4, 5: movement helps cube, hurts sponge',
       ['cube:'] + HOM + ['sponge:'] + HO, feats[HOM].values, per_object=True, X_sponge=feats[HO].values)

# %% Save
pd.DataFrame(results).to_csv(RESULTS_DIR / 'tree_results.csv', index=False)
print('Saved', RESULTS_DIR / 'tree_results.csv')
