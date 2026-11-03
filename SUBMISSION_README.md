# ELEC0144 Assignment Part I — Team 10

Image Processing and Neural Networks: Task 1, Task 2, the two What-if analyses and the Open Exploration.
Report: `ELEC0144_Team10_Report.pdf`. Self-assessment form: `SelfAssessment_Team10.pdf`.
Every file starts with a code map; its last line names the AI model used for that file (report, reference [1]).

## Setup

Python 3.10 or later. From this folder, either

```
pip install numpy pandas matplotlib opencv-python scikit-learn imageio-ffmpeg
```

or, with uv, `uv sync` (uses pyproject.toml and uv.lock) and then put `uv run` in front of each command below.
Task 1, Task 2 and the two What-if files need only NumPy, pandas and Matplotlib; OpenCV, scikit-learn and
imageio-ffmpeg are for the Open Exploration. Run every command from the folder it is listed for: all files read
their data with relative paths.

## Section A — Task 1 and What-if 1 (this folder)

| Command | What it does | Time |
|---|---|---|
| `python Task1.py` | Finds the apple, capsicum and lemon in Fruits.jpg. `darkFactor` and `mirror` are at the top of the file (submitted: 0.8 and True). Draws each box with both centroids and prints both centroids and the distance between them. | about 5 s |
| `python WhatIf_Task1.py` | What-if 1. Runs Task1.py unchanged except for its darkFactor line, for 1.0 down to 0.1 in steps of 0.05. Prints the table in the report (Table W1) and saves WhatIf_Task1.png (Figure W1). | about 1 min |

Task1.py prints, at darkFactor 0.8 with mirror = True:

```
fruit           box x       box y   box midpoint (x, y)    moments (x, y)  distance
capsicum   771-966     339-515    ( 868.5,  427.0)      ( 861.0,  430.8)     8.4 px
apple      549-777     436-560    ( 663.0,  498.0)      ( 654.3,  504.6)    10.9 px
lemon      300-458     203-300    ( 379.0,  251.5)      ( 380.9,  250.9)     2.0 px
```

## Section B — Task 2 and What-if 2 (this folder)

| Command | What it does | Time |
|---|---|---|
| `python Task2.py` | Trains the 1-3-1 network from InitialWeights.npz on RegressionData.csv, once with SGD and once with ADAM (3000 epochs each, seed 1). Plots the fit on `xtest` with the training data and both learning curves; prints the final training errors and the first epoch below 0.001. The comments in `forward`, `gradients` and `train` give the equation numbers of the derivation (report, Appendix A). | about 15 s |
| `python WhatIf_Task2.py` | What-if 2. Runs Task2.py with only `slope = 1 - a1**2` replaced by 1. Prints the numbers in the report and saves WhatIf_Task2.png (Figure W2). | about 40 s |

Task2.py prints:

```
Gradient check: largest |backprop - finite difference| over the 10 weights = 2.6e-11
Training error with the given weights (before training): 0.5170
SGD : final training error after 3000 epochs = 0.000475; first below 0.001: epoch 188
ADAM: final training error after 3000 epochs = 0.000373; first below 0.001: epoch 151
```

WhatIf_Task2.py prints, first:

```
Final training error after 3000 epochs, mean of (d - y)^2 over the 41 samples
         normal  slope = 1   ratio   first epoch below 0.001 (normal; slope = 1)
SGD    0.000475   0.028677   60.37   188; not reached
ADAM   0.000373   0.027383   73.50   151; not reached

Best straight line through the 41 samples (least squares): training error 0.0273
```

## Section C — Open Exploration (OpenExploration/)

As the brief says, this code is read, not re-run: the videos are not submitted. Every number and figure in the report
is saved in `OpenExploration/results/` by the files below. With the videos in a folder `open exploration videos/`
next to this README, the files run from the OpenExploration folder in this order (`python OE1_Grips.py`, and so on).
The first run decodes the videos into `OpenExploration/cache/` (about 1.3 GB, a few minutes).

| File | What it does | Writes to results/ |
|---|---|---|
| `oe_common.py` | Shared functions: read a video, hand mask (skin colour), cube or sponge clip, cut into grips, the measures of a grip. | — |
| `OE1_Grips.py` | Cuts the 8 videos into 160 grips, draws the test member by lot (seed 2026), and counts the grips under 45 other cutting settings. | grips.csv, counts.csv, cutting_robustness.csv, fig_grip_cutting.png, fig_all_grips.png |
| `OE2_SessionCheck.py` | Can frames with no hand in view tell the firm take from the light take? | session_check.csv, take_differences.csv |
| `OE3_Explore.py` | The attempts in the exploration tree, with leave-one-member-out validation on the training members. | tree_results.csv |
| `FinalMethod.py` | Trains the final method on the training members and runs it once on the test member (Yik Hui); compares with guessing. | final_test_result.csv, final_test_predictions.csv, fig_final_method_io.png |
| `OE4_Extension.py` | Going further B: ablations on the test member, and why the cube model fails. | extension_ablations.csv, per_person_cues.csv, why_cube.csv, cube_model_weights.csv, fig_extension.png |
| `OE5_SkinCheck.py` | The test result with each skin-colour bound moved (about 10 minutes). | skin_check.csv |

`results/00_written_before_exploring.md` holds the question and hypothesis, written before any method was tried;
`results/01_final_choice_before_test.md` holds the choice of final method, written before it was run on the test set.
