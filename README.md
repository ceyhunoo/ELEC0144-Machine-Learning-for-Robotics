# ELEC0144-Machine-Learning-for-Robotics

## Running the code

Run 
```
uv sync
```
If it fails, install uv first (Look online on how to do that)

Then, from this folder:
```
uv run python Task1.py
uv run python Task2.py
```
Both files read their data with relative paths, so run them from this folder. Task2.py takes about 15 seconds.

## SECTION A

### Further Explanation

Further explanation can be found here: https://claude.ai/artifact/J6W5fRfCkD38cVoaPGWb34

### How it works, and why each choice

1. **Darken and mirror.** The image is darkened with `floor(k·I + 0.5)` and then flipped.
   - The brief says "rounded", and this matches MATLAB's `round`.
   - `np.round` rounds .5 to the nearest even number instead, so it would differ on a few pixels.

2. **Rules that don't care about darkFactor.** Darkening multiplies R, G and B by the same k.
   - So every rule compares one channel with R, e.g. `B < 0.05·R`. The k appears on both sides and cancels.
   - Brightness is relative: each pixel's (R+G+B)/3 divided by the image mean. Both scale by k, so the ratio doesn't change.
   - This is why the results are identical from 0.6 to 1.0; only rounding differs.

3. **One colour rule per fruit:**
   - **Capsicum: almost no blue, and bright.**
     - In most of its pixels B is 0 and R is clipped at 255.
     - The wall has a little blue. Its dark shadow band has no blue either, but the brightness condition removes it.
   - **Apple: green well below red, blue nearly as high as green, and bright.**
     - The apple is a pinkish red.
     - The wall, bowl and capsicum have blue well below green.
     - The brightness condition removes the dark table and grape shadows.
   - **Lemon: green nearly equal to red (yellow), and little blue.**
     - The banana is also yellow but paler, with B about 0.6R, so the blue condition separates them.

4. **Median filter, 7×7.** On a 0/1 mask the median is a majority vote. It removes specks and streaks up to 3 pixels thick, cutting about 700 blobs down to about 75. The boxes barely change for any window from 1 to 11, so it is cleaning, not tuning.

5. **Labelling.** My own flood fill does this, because `bwlabel` is banned. It uses 8 neighbours and grows each blob from a seed pixel using a stack. The largest blob in each mask is taken as the fruit. This uses no coordinates, so mirroring can't break it. At every setting tested, the largest blob is at least 3× the next one.

6. **Measurement.**
   - The box comes from the min and max of x and y; its midpoint is the first centroid.
   - The moment centroid is M10/M00 and M01/M00, with I = 1 on the blob.
   - The moment centroid is an average of blob pixels, all of which are inside the box, so it can never fall outside it. The viva check "both centroids in every box" is therefore guaranteed.

**Why the two centroids differ.** The box midpoint uses only the four extreme pixels; the moment centroid uses every pixel.
- **Capsicum:** the mask has a hole at the green stem, which pulls the centroid away from it.
- **Apple:** the blob is a dome with a flat bottom along the bowl rim, so its mass sits lower than the box middle.
- **Lemon:** its blob is a nearly symmetric cap, so the two agree within 2 px.

### "How did you find this threshold?"

I took two steps:
1. **Starting values.** I plotted histograms of each ratio inside the fruit against the rest of the image.
2. **Working ranges.** I moved one threshold at a time and checked darkFactor 1.0, 0.8 mirrored and 0.6. I recorded the range over which all three fruits were still found correctly and picked a value near the middle.

The ranges are in the comments next to each value, for example capsicum `B < 0.05R` works from 0.005 to 0.10. That gives you a concrete answer at the viva.

To measure box coverage automatically, I drew rough outlines of the three fruits in my own checking script. That script is not part of Task1.py and shouldn't be submitted, because it contains hand-drawn coordinates. Your team should check the result by eye, the way the assessor will.

### Things to decide or check

- **Lemon coverage.** The lemon box covers about 76% of the visible lemon. The lower, more orange part has G/R around 0.5–0.7, the same as the wall.
  - Lowering `LEMON_G_MIN` to 0.6 raises coverage to about 89%, but that is the edge of its working range; at 0.55 it picks up the wall.
  - 0.75 is the safer choice, but it's the team's call.
- **Which fruit is the lemon.** I've taken it to be the big yellow citrus at the top. The yellow-green bits behind the grapes are not boxed.
- **AI line.** The code map's last line reads "Claude Opus 5.5 (claude.ai), default settings". Check the exact setting your app shows. If the team keeps the code unchanged, it also needs an IEEE reference in the report.
- **What-if 1.** I haven't run anything below 0.6. Each member's prediction comes first.

Next step: run it, read it line by line, and ask about anything you couldn't explain on your own. After that, the team reads it together and holds the What-if prediction meeting.

## SECTION B

### Further Explanation
Further explanation can be found here: https://claude.ai/artifact/DntJYyHRzgxuJMJ4YEuyRN

### What Task2.py does

It trains a 1-3-1 network (1 input, 3 hidden tanh nodes, 1 linear output) on the 41 samples in `RegressionData.csv`. It trains twice, both times from the starting weights in `InitialWeights.npz`:
- once with SGD (η = 0.1)
- once with ADAM (α = 0.005, β₁ = 0.9, β₂ = 0.999, ε = 1e-8)

Each run is 3000 epochs. It shuffles the samples every epoch and updates the weights after every sample. The network, backpropagation, SGD and ADAM are all written by hand with NumPy; pandas only reads the CSV.

### What you should see

Printed (seed 1):
```
Gradient check: largest |backprop - finite difference| over the 10 weights = 2.6e-11
Training error with the given weights (before training): 0.5170
SGD : final training error after 3000 epochs = 0.000475; first below 0.001: epoch 188
ADAM: final training error after 3000 epochs = 0.000373; first below 0.001: epoch 151
```

One figure with two panels:
- **Left:** the training data, with the network output on the 20 test inputs `xtest` for SGD (required) and ADAM. Both follow the data.
- **Right:** both learning curves on a log scale, with a line at 0.001 and a dot where each curve first goes below it.

These are what the assessor looks at during the viva: does the fit follow the data, do both curves come down and flatten, and is the final error printed.

### How it works, and why each choice

1. **Weights kept as matrices.** `W1` is 3×1, `b1` 3×1, `W2` 1×3, `b2` 1×1, as they come in the file.
   - With these shapes each line of code reads like the matrix equation in the derivation.
   - The four are kept together as `params = [W1, b1, W2, b2]`, so the update loops can go over all of them with `for k in range(4)`.

2. **Forward pass (`forward`).** For one input x:
   - `v1 = W1 * x + b1` gives the induced field of each hidden node (eq. 1).
   - `a1 = np.tanh(v1)` gives the hidden outputs, between −1 and 1 (eq. 2).
   - `y = W2 @ a1 + b2` gives the output: a weighted sum of the three tanh curves plus a bias (eq. 3).
   - The output node is linear because this is regression with an unconstrained target; the Week 2 slides pair that with a linear output. A sigmoid output could not even produce the negative values in the data.

3. **Backpropagation (`gradients`).** The idea: E depends on a weight only through a chain of quantities (weight → v → a → y → E). The derivative of E with respect to that weight is the product of the derivatives along the chain (the chain rule). Backprop computes the part of the chain that many weights share, the delta, once and reuses it.
   - **Output delta:** `delta_o = d - y` (eq. 5). The output node is linear, so its slope is 1 and the delta is just the error.
   - **Output-layer gradients:** `gW2 = -delta_o * a1.T` and `gb2 = -delta_o` (eqs. 6, 7). Each gradient is minus the delta times whatever arrives on that weight: a1 for W2, and 1 for the bias.
   - **Hidden delta:** `delta_h = W2.T * delta_o * slope`, with `slope = 1 - a1**2` (eqs. 8, 9). The output error is passed back through the weight that joins each hidden node to the output, then multiplied by the slope of tanh at that node. Since tanh′(v) = 1 − tanh²(v) = 1 − a², the slope comes straight from a1, which the forward pass already computed.
   - **Hidden-layer gradients:** `gW1 = -delta_h * x` and `gb1 = -delta_h` (eqs. 10, 11).
   - All gradients come from the same, old weights; only then are the weights changed. Updating W2 first and using the new W2 in the hidden delta would be a different algorithm.

4. **Gradient check.** Before training, every backprop gradient is compared with the slope measured directly.
   - Nudge one weight by ±1e-6, recompute E, and take (E(w + h) − E(w − h)) / 2h.
   - The two agree to 2.6e-11 for all 10 weights, so the formulas and the code match.
   - This is the answer to "how do you know your backprop is right?".

5. **Training loop (`train`).**
   - **Seed.** `rng = np.random.default_rng(seed)` runs at the start of every training run. SGD and ADAM therefore see exactly the same sample orders, and any other file using the same seed (the What-if file) gets them too, as the brief asks.
   - **Shuffle.** `order = rng.permutation(N)` gives a new order of the 41 samples every epoch. The file lists the samples sorted by x; without shuffling, every epoch would push the weights the same way at the same point.
   - **One update per sample,** as the brief requires.
   - **SGD (eq. 12):** `params[k] -= eta * g[k]`, a step downhill whose size is proportional to the gradient.
   - **ADAM (eqs. 13–15):** for each weight it keeps `m`, a running mean of the gradient, and `v`, a running mean of the squared gradient. Both start at zero, and so does the step count `t`.
     - `m_hat` and `v_hat` correct for that zero start. Early on m and v are too small; dividing by 1 − β^t scales them back up, and the correction fades as t grows.
     - The step is `alpha * m_hat / (np.sqrt(v_hat) + eps)`. Dividing by the root of the mean squared gradient makes the step roughly α for every weight, whatever the size of its gradient. `eps` only prevents division by zero.
     - `t` counts updates (one per sample), not epochs.

6. **Training error and first epoch below 0.001.**
   - After each epoch, `training_error` runs the network on all 41 samples and takes the mean of (d − y)². This is the brief's definition; it has no ½, unlike E.
   - `first_below` returns the first epoch, counting from 1, whose error is under 0.001, or "not reached".

### Derivation and the code

The lines of `forward`, `gradients` and `train` are labelled with these equation numbers in the code comments. The appendix should use the same numbers.

**Notation (Week 3 lecture).** A delta is the local gradient δ = −∂E/∂v of a node, the same sign as slides 53, 70 and 74. So every gradient is −δ × (input to that weight), and the SGD step w − η·∂E/∂w is the lecture's w + η·δ·x_out. The lecture's general hidden delta is (Σ_k δ_k·w_jk)·φ′(v_j); with one output node the sum has one term, which is eq. 9. The lecture calls the loss L; the brief and our code call it E. Our symbols map to the lecture's (layer s = 1 hidden, s = 2 output; w⁽ˢ⁾ᵢⱼ goes from node i to node j, i = 0 is the bias):

| Ours (code) | Lecture |
|---|---|
| x | x⁽⁰⁾_out,1 |
| `W1[j]`, `b1[j]` | w⁽¹⁾₁ⱼ, w⁽¹⁾₀ⱼ |
| `v1[j]`, `a1[j]` | v⁽¹⁾ⱼ, x⁽¹⁾_out,j |
| `W2[j]`, `b2` | w⁽²⁾ⱼ₁, w⁽²⁾₀₁ |
| v_o, y | v⁽²⁾₁, y₁ |
| `delta_o`, `delta_h[j]` | δ⁽²⁾₁, δ⁽¹⁾ⱼ (slide 74: δₒ, δₕ) |
| `slope[j]` | ∂φ⁽¹⁾/∂v⁽¹⁾ⱼ |

Code indices j run 0–2; the maths numbers the hidden nodes 1–3.

| Eq. | Maths | Code |
|---|---|---|
| 1 | v_j = W1_j·x + b1_j | `v1 = W1 * x + b1` |
| 2 | a_j = tanh(v_j) | `a1 = np.tanh(v1)` |
| 3 | y = v_o = Σ_j W2_j·a_j + b2 | `y = (W2 @ a1 + b2).item()` |
| 4 | E = ½(d − y)² | `gradients` docstring; used in the gradient check |
| 5 | δ_o = −∂E/∂v_o = d − y | `delta_o = d - y` |
| 6 | ∂E/∂W2_j = −δ_o·a_j | `gW2 = -delta_o * a1.T` |
| 7 | ∂E/∂b2 = −δ_o | `gb2 = -delta_o * np.ones((1, 1))` |
| 8 | tanh′(v_j) = 1 − a_j² | `slope = 1 - a1**2` |
| 9 | δ_j = −∂E/∂v_j = W2_j·δ_o·(1 − a_j²) | `delta_h = W2.T * delta_o * slope` |
| 10 | ∂E/∂W1_j = −δ_j·x | `gW1 = -delta_h * x` |
| 11 | ∂E/∂b1_j = −δ_j | `gb1 = -delta_h` |
| 12 | w ← w − η·∂E/∂w = w + η·δ·input | `params[k] -= eta * g[k]` |
| 13 | m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g² | `m[k] = ...`, `v[k] = ...` |
| 14 | m̂ = m / (1 − β₁ᵗ), v̂ = v / (1 − β₂ᵗ) | `m_hat = ...`, `v_hat = ...` |
| 15 | w ← w − α·m̂ / (√v̂ + ε) | `params[k] -= alpha * m_hat / (np.sqrt(v_hat) + eps)` |

The brief asks for the full calculation for one weight in each layer, then the general formula:
- **Output layer, W2_1.** The chain is E → y → W2_1, so ∂E/∂W2_1 = ∂E/∂y · ∂y/∂W2_1 = −(d − y)·a_1. Changing the index gives eqs. 6 and 7.
- **Hidden layer, W1_1.** The chain is E → y → a_1 → v_1 → W1_1, so ∂E/∂W1_1 = −(d − y)·W2_1·(1 − a_1²)·x. Changing the index gives eqs. 10 and 11.

### Worked step (to check by hand)

At x = 0.5 (d = −0.0521) with the given weights:
- **Forward:** v = (2, −0.75, −0.5), a = (0.9640, −0.6351, −0.4621), y = 0.5685.
- **Output:** δ_o = −0.0521 − 0.5685 = −0.6206. ∂E/∂W2_1 = 0.6206 × 0.9640 = 0.5983, so one SGD step takes W2_1 from 0.5 to 0.5 − 0.1 × 0.5983 = 0.4402.
- **Hidden node 1:** slope = 1 − 0.9640² = 0.0707, so δ_1 = 0.5 × (−0.6206) × 0.0707 = −0.0219. Then ∂E/∂W1_1 = 0.0219 × 0.5 = 0.0110, so W1_1 goes from 2 to 1.9989.

### Reading the results

- **The curves jitter.** The weights change after every sample with a fixed step, so they never settle; they keep moving around the minimum. The error recorded at the end of an epoch depends on the last few samples, so it bounces from one epoch to the next. Both curves still come down and flatten.
- **Where they flatten.** Both settle around 0.0003–0.0004. The data were made with noise of standard deviation 0.02, whose variance is 0.02² = 0.0004. So the network fits the underlying curve to about the level of the noise.
- **First epoch below 0.001.** With seed 1, ADAM gets there at epoch 151 and SGD at 188.
  - With seeds 1–8, ADAM is at 141–155 and SGD at 184–194, so the order is not a lucky shuffle.
  - ADAM's final error varies more with the seed, from 0.0003 to 0.0012; seed 3 ends above 0.001.
  - Why ADAM gets there earlier is a likely viva question. The explanation will be added here once it has been checked.

### Things to decide or check

- **Notation.** Checked against the Week 3 lecture: same sign convention (δ = −∂E/∂v), mapping in the table above.
- **ADAM form.** This is the standard version (Kingma and Ba), with bias correction and ε outside the square root. Check it against the lecture that covers ADAM.
- **Two extras beyond the brief.** The Gradient check section, and ADAM's fit on the left panel (the brief only asks for SGD's). Each is one more thing someone can be asked about. Keep or drop them.
- **Derivation appendix.** Required, at most 2 pages; a scan of handwritten work is fine. Use the equation numbers above.
- **AI line.** Same as Task 1: "Claude Opus 5.5 (claude.ai), default settings". Check the exact setting, and add an IEEE reference if the code is kept unchanged.
- **What-if 2.** The factor to replace is `slope = 1 - a1**2` (eq. 8) in `gradients`. Nothing has been run with it changed. Each member writes their name, prediction and reason first. Then the change goes in a separate file, WhatIf_Task2.py.

Next step: run it, read it next to the table above, and ask about any line you couldn't explain on your own. Then the team writes the appendix after the Week 3 lecture and holds the What-if 2 prediction meeting.
