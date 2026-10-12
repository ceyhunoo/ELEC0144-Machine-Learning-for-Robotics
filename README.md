# ELEC0144-Machine-Learning-for-Robotics

## SECTION A

## How it works, and why each choice

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

## "How did you find this threshold?"

I took two steps:
1. **Starting values.** I plotted histograms of each ratio inside the fruit against the rest of the image.
2. **Working ranges.** I moved one threshold at a time and checked darkFactor 1.0, 0.8 mirrored and 0.6. I recorded the range over which all three fruits were still found correctly and picked a value near the middle.

The ranges are in the comments next to each value, for example capsicum `B < 0.05R` works from 0.005 to 0.10. That gives you a concrete answer at the viva.

To measure box coverage automatically, I drew rough outlines of the three fruits in my own checking script. That script is not part of Task1.py and shouldn't be submitted, because it contains hand-drawn coordinates. Your team should check the result by eye, the way the assessor will.

## Things to decide or check

- **Lemon coverage.** The lemon box covers about 76% of the visible lemon. The lower, more orange part has G/R around 0.5–0.7, the same as the wall.
  - Lowering `LEMON_G_MIN` to 0.6 raises coverage to about 89%, but that is the edge of its working range; at 0.55 it picks up the wall.
  - 0.75 is the safer choice, but it's the team's call.
- **Which fruit is the lemon.** I've taken it to be the big yellow citrus at the top. The yellow-green bits behind the grapes are not boxed.
- **AI line.** The code map's last line reads "Claude Opus 5.5 (claude.ai), default settings". Check the exact setting your app shows. If the team keeps the code unchanged, it also needs an IEEE reference in the report.
- **What-if 1.** I haven't run anything below 0.6. Each member's prediction comes first.

Next step: run it, read it line by line, and ask about anything you couldn't explain on your own. After that, the team reads it together and holds the What-if prediction meeting.

Status updated: Task 1 v1 recorded with method, thresholds and ranges; code saved to claude/code/Task1.py; Python noted as a working assumption; session log updated.
