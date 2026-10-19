# CODE MAP of Task2.py
# Settings        file names, epochs, random seed, SGD learning rate, ADAM constants
# Load data       read RegressionData.csv and InitialWeights.npz, make xtest
# Network         forward (hidden outputs a1 and output y for one input), gradients (output delta,
#                 hidden delta, dE/dw for every weight of one sample), predict, training_error
# Gradient check  compare the backprop gradients with finite differences at the given weights
# Training        train(): shuffle every epoch, update after every sample with SGD or ADAM,
#                 record the training error at the end of each epoch
# Train SGD       train with SGD from the given weights
# Train ADAM      train with ADAM from the given weights
# Print results   initial and final training errors, first epoch below 0.001
# Plots           network output on xtest with the training data; both learning curves
# AI use          Claude Opus 5.5 (claude.ai), default settings

# %% Settings
# Run from the folder that contains RegressionData.csv and InitialWeights.npz:  python Task2.py
# Equation numbers (1)-(15) refer to the derivation in the report appendix.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = 'RegressionData.csv'
WEIGHTS_FILE = 'InitialWeights.npz'
n_epochs = 3000           # passes through the 41 samples
seed = 1                  # set at the start of every training run -> same shuffle order in any file
eta = 0.1                 # SGD learning rate
alpha = 0.005             # ADAM step size
beta1 = 0.9               # ADAM decay rate of m (running mean of the gradient)
beta2 = 0.999             # ADAM decay rate of v (running mean of the squared gradient)
eps = 1e-8                # ADAM: stops division by zero when v is ~0
target = 0.001            # report the first epoch whose training error is below this


# %% Load data
data = pd.read_csv(DATA_FILE)
x_train = data['x'].to_numpy()        # 41 inputs, -1 to 1 in steps of 0.05
d_train = data['d'].to_numpy()        # 41 desired outputs (cubic + noise)
N = len(x_train)

w0 = np.load(WEIGHTS_FILE)
W1r, b1r = w0['W1r'], w0['b1r']       # input -> hidden weights and hidden biases, both 3 x 1
W2r, b2r = w0['W2r'], w0['b2r']       # hidden -> output weights 1 x 3, output bias 1 x 1

xtest = np.linspace(-0.97, 0.93, 20)  # 20 test inputs, not in the training set


# %% Network
def forward(x, W1, b1, W2, b2):
    """Forward pass for one input.
    In:  x, a number; W1, b1 (3 x 1); W2 (1 x 3); b2 (1 x 1).
    Out: a1 (3 x 1), outputs of the three hidden tanh nodes; y, the network output (a number)."""
    v1 = W1 * x + b1                  # (1) induced field of each hidden node
    a1 = np.tanh(v1)                  # (2) hidden outputs, between -1 and 1
    y = (W2 @ a1 + b2).item()         # (3) linear output node: weighted sum of a1 plus bias
    return a1, y


def gradients(x, d, W1, b1, W2, b2):
    """Backpropagation for one sample (x, d), with E = 1/2 (d - y)^2   (4).
    In:  one sample and the current weights.
    Out: [dE/dW1, dE/db1, dE/dW2, dE/db2], each the same shape as its weight."""
    a1, y = forward(x, W1, b1, W2, b2)
    delta_o = d - y                     # (5) output delta: error times slope of the linear output (= 1)
    gW2 = -delta_o * a1.T               # (6) dE/dW2_j = -delta_o * a1_j          (1 x 3)
    gb2 = -delta_o * np.ones((1, 1))    # (7) dE/db2   = -delta_o                 (1 x 1)
    slope = 1 - a1**2                   # (8) slope of tanh at each hidden node   (3 x 1)
    delta_h = W2.T * delta_o * slope    # (9) hidden delta: error passed back through W2, times slope
    gW1 = -delta_h * x                  # (10) dE/dW1_j = -delta_h_j * x          (3 x 1)
    gb1 = -delta_h                      # (11) dE/db1_j = -delta_h_j              (3 x 1)
    return [gW1, gb1, gW2, gb2]


def predict(xs, params):
    """Network output for every value in xs (array). params = [W1, b1, W2, b2]."""
    return np.array([forward(x, *params)[1] for x in xs])


def training_error(params):
    """Mean of (d - y)^2 over the 41 training samples."""
    return np.mean((d_train - predict(x_train, params)) ** 2)


# %% Gradient check
# Backprop gives dE/dw from formulas. Check each one against the slope measured directly:
# nudge that one weight by +h and -h, recompute E, slope = (E(w + h) - E(w - h)) / 2h.
# Done once at the given weights for one training sample; the two should agree to ~1e-9.
h = 1e-6
xs, ds = x_train[30], d_train[30]
params0 = [W1r, b1r, W2r, b2r]
analytic = gradients(xs, ds, *params0)
max_diff = 0.0
for k in range(4):
    for idx in np.ndindex(params0[k].shape):
        plus = [p.copy() for p in params0]
        minus = [p.copy() for p in params0]
        plus[k][idx] += h
        minus[k][idx] -= h
        E_plus = 0.5 * (ds - forward(xs, *plus)[1]) ** 2
        E_minus = 0.5 * (ds - forward(xs, *minus)[1]) ** 2
        numeric = (E_plus - E_minus) / (2 * h)
        max_diff = max(max_diff, abs(numeric - analytic[k][idx]))
print(f'Gradient check: largest |backprop - finite difference| over the 10 weights = {max_diff:.1e}')


# %% Training
def train(method):
    """Train the 1-3-1 network from the given weights with method 'SGD' or 'ADAM'.
    Every epoch: shuffle the 41 samples, then update the weights after each sample.
    Out: params, the final [W1, b1, W2, b2]; err, the training error after each epoch."""
    rng = np.random.default_rng(seed)                  # same seed at the start of every run
    params = [W1r.copy(), b1r.copy(), W2r.copy(), b2r.copy()]
    m = [np.zeros_like(p) for p in params]             # ADAM: running mean of each gradient
    v = [np.zeros_like(p) for p in params]             # ADAM: running mean of each gradient squared
    t = 0                                              # ADAM: number of updates so far
    err = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        order = rng.permutation(N)                     # shuffle: a new random order of the samples
        for i in order:
            g = gradients(x_train[i], d_train[i], *params)
            if method == 'SGD':
                for k in range(4):
                    params[k] -= eta * g[k]                                  # (12) SGD step
            else:
                t += 1
                for k in range(4):
                    m[k] = beta1 * m[k] + (1 - beta1) * g[k]                 # (13) running means
                    v[k] = beta2 * v[k] + (1 - beta2) * g[k] ** 2
                    m_hat = m[k] / (1 - beta1 ** t)                          # (14) bias correction
                    v_hat = v[k] / (1 - beta2 ** t)
                    params[k] -= alpha * m_hat / (np.sqrt(v_hat) + eps)      # (15) ADAM step
        err[epoch] = training_error(params)            # end of epoch: error over all 41 samples
    return params, err


# %% Train SGD
params_sgd, err_sgd = train('SGD')


# %% Train ADAM
params_adam, err_adam = train('ADAM')


# %% Print results
def first_below(err):
    """First epoch (counting from 1) whose training error is below target, or 'not reached'."""
    hits = np.nonzero(err < target)[0]
    return str(hits[0] + 1) if hits.size else 'not reached'


print(f'Training error with the given weights (before training): {training_error(params0):.4f}')
print(f'SGD : final training error after {n_epochs} epochs = {err_sgd[-1]:.6f}; '
      f'first below {target}: epoch {first_below(err_sgd)}')
print(f'ADAM: final training error after {n_epochs} epochs = {err_adam[-1]:.6f}; '
      f'first below {target}: epoch {first_below(err_adam)}')


# %% Plots
epochs = np.arange(1, n_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

ax1.plot(x_train, d_train, 'o', color='grey', label='training data (41 samples)')
ax1.plot(xtest, predict(xtest, params_sgd), 's-', color='tab:blue', label='SGD network output on xtest')
ax1.plot(xtest, predict(xtest, params_adam), '^--', color='tab:orange', mfc='none',
         label='ADAM network output on xtest')
ax1.set_xlabel('input x')
ax1.set_ylabel('output')
ax1.set_title('1-3-1 network after 3000 epochs')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.semilogy(epochs, err_adam, color='tab:orange', lw=0.8, label=f'ADAM (alpha = {alpha})')
ax2.semilogy(epochs, err_sgd, color='tab:blue', lw=0.8, label=f'SGD (eta = {eta})')
ax2.axhline(target, color='k', linestyle=':', label=f'{target}')
for err, colour in [(err_sgd, 'tab:blue'), (err_adam, 'tab:orange')]:
    first = first_below(err)
    if first != 'not reached':                       # mark the printed epoch on its curve
        ax2.plot(int(first), err[int(first) - 1], 'o', color=colour, mec='k', ms=7)
ax2.set_xlabel('epoch')
ax2.set_ylabel('training error, mean of (d - y)$^2$')
ax2.set_title('Learning curves')
ax2.legend()
ax2.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()
