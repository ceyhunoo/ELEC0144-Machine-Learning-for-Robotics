# Final method chosen before it was run on the test set (Sat 31 Oct 2026, 19:56)

(Time corrected on 31 Oct from the file's own timestamp, 19:55:56 GMT; the first version said
"~20:25" by mistake. Nothing else in this note was changed.)

Chosen: attempt 8 (validation 86.7%: cube 76.7%, sponge 96.7%; leave one member out over Ceyhun, K, Shangze).
- Cube model: hand area + object area/shape + movement (13 measures).
- Sponge model: hand area + object area/shape (8 measures).
- Standardise, then logistic regression (C = 1), trained on all 120 training grips.

Note written at the same time: attempt 5 (84.2%) differs by 3 of 120 grips, within noise
(standard error of a validation accuracy near 85% with 120 grips is about 3 points).
Attempt 8 was chosen as the best so far, as the brief asks.

Test set: all 40 grips of Yikhui (Yikhui firm.mp4, Yikhui gentle.mp4), drawn by lot (seed 2026) before exploring.
