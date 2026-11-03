# CODE MAP of WhatIf_Task2.py  (What-if 2: the forgotten derivative; everything else as in Task2.py)
# Settings        the one line of Task2.py that is changed, and what it is changed to
# Run Task2       run_task2: runs Task2.py up to its plots, as it is or with the slope of tanh
#                 replaced by 1; returns Task2.py's variables (weights, learning curves, functions)
# Run both        the normal run and the run with the slope replaced by 1 (SGD and ADAM in each)
# Print           final training error, ratio to the normal run, first epoch below 0.001 (the
#                 numbers in the report); for slope = 1: lowest error and its epoch, mean of the
#                 last 500 epochs, the best straight line's error, final weights and hidden inputs
# Figure          fits on xtest (SGD, ADAM) and the four learning curves (WhatIf_Task2.png)
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the folder that contains Task2.py, RegressionData.csv and InitialWeights.npz:
#   python WhatIf_Task2.py        (about 40 seconds)
# Task2.py is not edited: its text is read and run twice, once as it is and once with the line
# of eq. (8) changed. Same data, starting weights, seed, settings and update rules in both runs;
# Task2.py's train() starts m, v and t at zero and sets the seed at the start of every run.
import contextlib
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')                       # no windows; the figure is saved to a file
import matplotlib.pyplot as plt

TASK2_SOURCE = open('Task2.py').read().split('# %% Plots')[0]    # everything except Task2's plots
NORMAL_LINE = 'slope = 1 - a1**2'           # eq. (8): slope of tanh at each hidden node, 1 - a^2
FORGOTTEN_LINE = 'slope = np.ones_like(a1)'  # the What-if: the slope is replaced by 1
assert TASK2_SOURCE.count(NORMAL_LINE) == 1  # the one line that is changed


# %% Run Task2
def run_task2(forget_slope: bool) -> dict:
    """Run Task2.py without its plots. forget_slope=True replaces 1 - a1**2 by 1 in gradients().
    Returns Task2.py's variables, e.g. params_sgd, err_sgd, params_adam, err_adam, predict."""
    code = TASK2_SOURCE.replace(NORMAL_LINE, FORGOTTEN_LINE) if forget_slope else TASK2_SOURCE
    names = {}
    with contextlib.redirect_stdout(io.StringIO()):            # Task2.py's own printout is hidden
        exec(compile(code, 'Task2.py', 'exec'), names)
    return names


# %% Run both
normal = run_task2(forget_slope=False)
forgot = run_task2(forget_slope=True)


# %% Print
first_below = normal['first_below']          # Task2.py: first epoch below 0.001, or 'not reached'
x, d = normal['x_train'], normal['d_train']
line_err = np.mean((d - np.polyval(np.polyfit(x, d, 1), x)) ** 2)    # best straight line y = p*x + q

print('Final training error after 3000 epochs, mean of (d - y)^2 over the 41 samples')
print(f'{"":5s} {"normal":>9s} {"slope = 1":>10s} {"ratio":>7s}   first epoch below 0.001 (normal; slope = 1)')
for m in ('sgd', 'adam'):
    e0, e1 = normal[f'err_{m}'], forgot[f'err_{m}']
    print(f'{m.upper():5s} {e0[-1]:9.6f} {e1[-1]:10.6f} {e1[-1] / e0[-1]:7.2f}   '
          f'{first_below(e0)}; {first_below(e1)}')

print(f'\nBest straight line through the 41 samples (least squares): training error {line_err:.4f}')
print('With slope = 1:')
for m in ('sgd', 'adam'):
    e1 = forgot[f'err_{m}']
    i = int(e1.argmin())
    W1, b1, W2, b2 = forgot[f'params_{m}']
    v = W1 * x + b1                                             # hidden inputs for all 41 samples
    print(f'  {m.upper():4s} lowest error {e1[i]:.4f} at epoch {i + 1}; '
          f'mean of the last 500 epochs {e1[-500:].mean():.4f}')
    print(f'       final W1 = {np.round(W1.ravel(), 3)}, W2 = {np.round(W2.ravel(), 3)}')
    print('       hidden input range of each node over the data: '
          + ', '.join(f'{v[j].min():.2f} to {v[j].max():.2f}' for j in range(3)))


# %% Figure
# Sized for the report (about 16 cm wide), so the text prints at about 6-7 pt.
plt.rcParams.update({'font.size': 7})
xtest = normal['xtest']
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for ax, m, title in ((axs[0], 'sgd', '(a) SGD: fit on xtest'), (axs[1], 'adam', '(b) ADAM: fit on xtest')):
    ax.plot(x, d, 'o', color='grey', ms=2.2, label='training data')
    ax.plot(xtest, normal['predict'](xtest, normal[f'params_{m}']), 's-', color='tab:blue',
            ms=2.2, lw=0.9, label='normal')
    ax.plot(xtest, forgot['predict'](xtest, forgot[f'params_{m}']), '^--', color='tab:red',
            ms=2.2, lw=0.9, label='slope = 1')
    ax.set_xlabel('input x'); ax.set_ylabel('output')
    ax.set_title(title, fontsize=7); ax.legend(fontsize=5.5, loc='upper left')
ax = axs[2]
epochs = np.arange(1, len(normal['err_sgd']) + 1)
for m, c in (('sgd', 'tab:blue'), ('adam', 'tab:orange')):
    ax.semilogy(epochs, normal[f'err_{m}'], color=c, lw=0.6, label=f'{m.upper()}, normal')
    ax.semilogy(epochs, forgot[f'err_{m}'], color=c, lw=0.6, ls='--', label=f'{m.upper()}, slope = 1')
ax.axhline(line_err, color='k', lw=0.7, ls=':', label='best straight line')
ax.set_xlabel('epoch'); ax.set_ylabel('training error')
ax.set_ylim(2e-4, 3)                                            # room for the legend above the curves
ax.set_title('(c) Learning curves', fontsize=7); ax.legend(fontsize=5.3, loc='upper right', ncol=2, columnspacing=0.8)
fig.tight_layout()
fig.savefig('WhatIf_Task2.png', dpi=300)
print('\nFigure saved: WhatIf_Task2.png')
