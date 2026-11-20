# BOID for JEPA

Counting the oscillatory modes that a JEPA world-model latent sustains in time, under three training objectives: stop-gradient, SVReg and Slow SVReg.

Alexandra Bernadotte and Ivan Menshikov, Aicumene, November 2026. Status: a toy-scale first measurement on the stop-gradient model only; the SVReg and Slow SVReg regimes are left for the real model of Ren et al., not for a toy.

## What this is

Ren, Wang, Hoang, Kuang and LeCun (2026) show that a JEPA world model trained only to predict first-person video learns a latent map of a maze once sliced variance regularization (SVReg) holds a floor on the latent's spread along random directions, and that a slowness term with a gated residual predictor (Slow SVReg) keeps slowly changing factors that plain SVReg loses: the content–geometry tradeoff. Those are statements about the geometry of the latent at a moment.

BOID, the Bernadotte Oscillator Intrinsic Dimension, reads the same latent in time: it counts how many oscillatory modes a multichannel trajectory sustains, with a distribution-free guarantee from a block-Hankel embedding (multichannel singular spectrum analysis). This repository asks three questions of JEPA latents:

1. Is a regularized latent richer or poorer in time than a collapsed one?
2. Do the slow factors that plain SVReg loses reappear as low-frequency modes in the slow channels of Slow SVReg?
3. Does a world model have regimes when its predictor runs beyond the rollout horizon?

The code of Ren et al. is not released, so `jepa_boid_toy.py` builds a toy copy of their top-down PointMaze setup: a U-maze with a point agent and a pushable block (the slow factor), one uniform-random exploration trajectory of 5000 frames at 48×48 pixels, a convolutional encoder with BatchNorm, a predictor with a three-frame history and the action, latent dimension 32, and the three objectives written from the report's equations (1)–(3): SG, SVReg with λ = 10, Slow SVReg with the last 1/8 of the latent as slow channels, weight 1/λ, and the gated residual predictor with the gate initialised at 0.1. 3000 steps of batch 32. Only the stop-gradient model was run (one seed); the script contains all three objectives.

Measured on held-out trajectories, at three levels: the latent dimension (Hankel components above the 95 % permutation-surrogate threshold), the oscillatory dimension or BOID (resolved ESPRIT pairs at that rank, their stability across lag and record length, and the pairs above a red-noise null), and the autonomous regime (a 300-step open-loop rollout of the predictor, with zero actions and with the real action sequence, against a teacher-forced one-step baseline); with them the share of latent power at periods under five frames (jitter), the jaggedness of the latent path and the linear readout R² of the agent and block positions. Controls: the same encoder untrained (three initialisations) and the frames themselves in 32 principal components.

## Results (stop-gradient, maze2d-medium layout, three held-out runs of 1024 frames)

Three levels are kept apart, because a count of components above a surrogate threshold is a dimension and not a number of oscillators (Table 8 of the paper, Section 7.9; all numbers in `medium_SG_poles.json`, `medium_SG_checks.json`, `medium_SG_regimes.json`, `medium_SG_figure_counts.json`).

| quantity | run 0 | run 1 | run 2 |
|---|---|---|---|
| **Latent dimension: Hankel components above the permutation-surrogate threshold** | | | |
| trained encoder, whole run | 23 | 22 | 20 |
| in 256-frame windows | 11–19 | 9–21 | 12–18 |
| correlation of the windowed value with the agent's speed | +0.66 | +0.48 | +0.21 |
| correlation with the share of frames stuck at a wall | -0.69 | -0.56 | -0.67 |
| untrained encoder, three initialisations | 42–49 | 63–65 | 54–55 |
| the frames themselves, 32 principal components | 73 | 81 | 77 |
| **Oscillatory dimension (BOID): ESPRIT pairs at the Hankel rank** | | | |
| resolved pairs (period ≤ 512 frames) | 9 | 9 | 9 |
| unresolved pairs (period > 512 frames); real poles | 1; 3 | 1; 2 | 1; 0 |
| resolved pairs stable across lag 8/16/32 and length 512/768/1024 | 0 | 1 | 0 |
| components above the red-noise (AR(1)) null | 8 | 3 | 4 |
| resolved pairs among them, and their periods in frames | 3 (51, 15, 8) | 1 (84) | 1 (24) |
| resolved pairs above a 1/f^χ null | 2 | 0 | 0 |
| untrained encoder, pairs above the red-noise null | 3–5 | 4–7 | 3–5 |
| **Autonomous regime: 300-step open-loop rollout of the predictor** | | | |
| Hankel components at the threshold, zero actions / real action sequence | 1–2 / 1 | 1 / 1 | 1 / 1 |
| poles at rank 2 | pair, T = 1538 | pair, T = 1392 | real, |z| = 0.996 and 0.992 |
| step length, first 50 → last 50 steps | 0.012 → 0.0004 | 0.015 → 0.0006 | 0.022 → 0.0009 |
| end point, distance from the nearest data latent (cloud radii) | 2.5 | 1.3 | 2.7 |
| one-step prediction from the real history: Hankel components | 10 | 8 | 8 |
| **Content and geometry** | | | |
| share of latent power at periods under 5 frames | 0.079 | 0.044 | 0.046 |
| jaggedness of the latent path (1 = smooth) | 4.15 | 4.22 | 3.82 |
| readout R², agent position (ridge, 5-fold) | 0.16 | 0.62 | 0.58 |
| readout R², block position | 0.51 | not pushed | 0.74 |

**Three numbers, not one.** A count of components above a surrogate threshold is a dimension, not a number of oscillators: a two-dimensional subspace can carry one rotating pair, or two real poles, a trend and a damped relaxation. Table 8 therefore keeps three levels apart. The *latent dimension* is the number of components of the block-Hankel Gram matrix (lag 16) above the 95 % permutation-surrogate threshold, the surrogate-thresholded Hankel dimension of Section 3.2, with 200 surrogates. The *oscillatory dimension*, BOID, is the number of positive-frequency TLS-ESPRIT poles fitted at that rank with |z| ≥ 0.95 and a period of at most half the record, so that two cycles are observed; a pole with a longer period cannot be told from a trend and is listed apart, as are the real poles. Two checks follow it: whether a pair is found again, within 20 % of its period, at lags 8, 16 and 32 and record lengths 512, 768 and 1024, and whether it stands above an aperiodic null, the Monte-Carlo SSA index of Section 3.2 with red noise, an AR(1) process fitted per channel, as the null. The *autonomous regime* is what the predictor sustains when it is run on its own.

**The latent dimension.** On three held-out runs of 1024 frames the trained latent has 23, 22 and 20 Hankel components above the threshold. The untrained encoder on the same frames has 42–49, 63–65 and 54–55, and the frames themselves, in their 32 leading principal components, 73, 81 and 77: training reduces the temporal dimension of the latent, as an objective that rewards predictability should. The latent path is jagged, 4.1 times the length of its smoothed version where 1 is smooth, against 5.3 reported for the stop-gradient model of Ren et al. (2026); a cross-validated ridge readout recovers the agent's position with R² = 0.46 and, in the two runs in which the block was pushed, the block's with R² = 0.51 and 0.74.

**The oscillatory dimension.** At the Hankel rank ESPRIT finds 9 resolved pairs in each run, one pair with a period longer than the record (599, 553 and 1596 frames; the periods above 1000 frames in the first version of this table were of this kind and are dropped) and 3, 2 and 0 real poles with |z| between 0.98 and 1.00, the trend. Of the resolved pairs, 0, 1 and 0 are found again in all nine combinations of lag and record length; against the red-noise null 8, 3 and 4 components stand, and among them 3, 1 and 1 resolved pairs, at periods of 51, 15, 8, 84 and 24 frames; against a 1/f^χ null fitted per channel the pairs are 2, 0 and 0. Removing the linear trend and a 257-frame moving average from each channel leaves 8, 9 and 7 of the nine resolved pairs in place, so they are not an artefact of the drift, but they move with the lag, which is what the parametric decomposition of a broadband signal does. The untrained encoder has 3–5, 4–7 and 3–5 pairs above the red-noise null, the frames 70, 68 and 74. So the trained latent has a Hankel dimension above twenty and a BOID of one to three: a broadband, driven signal with few and weak rhythms, none of them robust to the embedding. In the terms of Section 2.4 the two levels differ here by an order of magnitude.

**When the latent dimension changes.** In 256-frame windows stepped by 32 frames the latent dimension moves between 9 and 21 (Figure 13), and in all three runs it follows how freely the agent moves: its correlation with the mean speed of the agent is +0.66, +0.48 and +0.21, with the share of frames in which the agent is stuck at a wall -0.69, -0.56 and -0.67; windows in which the agent is stuck more often than the run's median share hold on average 13.4, 13.2 and 13.4 components, the other windows 16.2, 15.6 and 15.5. Whether the agent is in a corridor or in a room does not decide it (the correlation with the share of frames in rooms is -0.75, +0.44 and +0.28), nor does contact with the block. Nor is it simply the Hankel dimension of the frames, which lies between 14 and 49 in the same windows and correlates with the latent's inconsistently, at +0.63, -0.03 and +0.34. The regimes of this latent are set by the behaviour of the agent through the encoder: a free run is a richer input than a run along a wall, and the latent carries that difference rather than the dimension of the pixels as such. BOID in the same windows stays between 0 and 4.

**On its own.** Run for 300 steps from a real three-frame history with the action held at zero, the predictor's latent has one or two Hankel components above the threshold (1, 1 and 1 in the three rollouts with one set of 200 permutation surrogates, 2 in the windows of Figure 13 with another: the second component sits at the threshold), and its poles at rank 2 are pair, T = 1538 (longer than the rollout), pair, T = 1392 (longer than the rollout) and real, |z| = 0.996 and 0.992: a trend, not an oscillation. The step shrinks from 0.012, 0.015 and 0.022 in the first fifty steps to 0.0004, 0.0006 and 0.0009 in the last fifty, and the trajectory converges to a point 2.5, 1.3 and 2.7 cloud radii from the nearest held-out latent, where the latents' own nearest-neighbour distance is 7 % of a radius: a fixed point outside the data region, the relaxation of Section 7.7. Driven by the real action sequence with its own latent history the result is the same, one component, real or unresolved poles, and an end point 2.4, 1.3 and 2.6 radii outside; so the zero action is not the cause. One step ahead, from the real history and the real action, the predictor is accurate (relative error below 10⁻³) and its output sustains 10, 8 and 8 components. The statement the experiment supports is therefore this: the tested stop-gradient model did not sustain a stable, structured autonomous regime under the evaluated open-loop rollout. Whether that is the absence of autonomous regimes or the accumulation of one-step errors in an autoregressive rollout, the experiment cannot separate; the predictor was trained on real histories and real actions, and a rollout on its own predictions leaves that distribution at the first step. A stronger conclusion needs a model trained with rollouts in the loop, or a closed-loop test in which the rollout is corrected by the encoder at intervals, so that the two causes come apart.

**What the three levels say together.** Not every rich latent representation has an autonomous dynamics, and not every oscillatory dynamics is functional. The toy latent is rich at the first level, with more Hankel components than the working regime of the CTM, nearly empty at the second, and empty at the third as tested; the CTM is the opposite case, a small number of rhythms above the background that persist beyond the horizon, switch with the stage of the computation and carry functions that ablation can remove (Sections 4–5). A single count, at either end, would have put the two systems on one axis. This is the conclusion of Section 8 made concrete: intelligence-like regimes are told apart by the combination of dimension, stability of the modes, controlled switching and the causal role of the modes, and no one of these alone. In the table of signatures the toy is a driven stream with input and a relaxation without it.

`jepa_boid_latent_3d.png` and `jepa_boid_latent_3d.mp4`: the latent in PC1–PC3 coloured by the agent's position, the maze with the same colours, and, along the run and into the rollout, the latent dimension (200 permutation surrogates) and BOID proper (resolved ESPRIT pairs above a red-noise null). The readout is a cross-validated ridge regression on the standardised latent (a split-halves readout fails on these runs because the agent visits different rooms in the two halves). An earlier U-maze version of the toy (`run_umaze/` in the working folder) gave 16–19 components, jaggedness 4.4 and the same collapse on the rollout.

We stopped at the stop-gradient model on purpose: the two regularized regimes should be measured on the released model, where the question is real, rather than on a toy.

## Run

```
pip install numpy torch matplotlib
python jepa_boid_toy.py 3000 3      # steps, seeds; writes jepa_boid_results.json, latents_*.npz, model_*.pt
python jepa_boid_figure.py          # latent maps, counts, spectra, readouts
python jepa_boid_poles.py           # three levels: Hankel dimension, ESPRIT pairs (resolved/unresolved/real), stability, MC-SSA, trend null
python jepa_boid_checks.py          # controls: red-noise null, rollout with real actions, teacher forcing, untrained encoder, pixel input
python jepa_boid_checks2.py         # red-noise null for the controls; windowed counts with 200 surrogates
python jepa_boid_3d.py              # 3-D latent map with the latent-dimension and BOID traces, PNG and rotating MP4
```

## References

- Ren M., Wang Y., Hoang C., Kuang Y., LeCun Y. (2026). The content–geometry tradeoff in world models: when JEPA learns latent maps. AGL-TR-2026-04. https://doi.org/10.5281/zenodo.24356797 · alphaXiv: https://www.alphaxiv.org/abs/2611.jepa-latent-maps-content-geometry · project page: https://agenticlearning.ai/content-geometry-tradeoff/
- Bernadotte A., Menshikov I. (2026). From Brain Dynamics to Strong Artificial Intelligence: Learning Creates Functional Regime Switching. Aicumene Preprint 2026.045v2. https://doi.org/10.13140/RG.2.2.22357.59364 · https://research.aicumene.com/papers/BOID-JEPA/
- Bernadotte A., Menshikov I. (2026). Multichannel Singular Spectrum Analysis: Counting Oscillatory Modes and Recovering Their Spatial Loadings. SSRN 7693202. https://doi.org/10.2139/ssrn.7693202 · code https://github.com/aicumene/MSSA
- Bernadotte A., Menshikov I. (2026). Brain Oscillator Intrinsic Dimension Marks Consciousness, Recovery Mode, and Failed Downshifting in Coma. SSRN 7667278. https://doi.org/10.2139/ssrn.7667278
- Bernadotte A. (2026). Topology-driven classification of time series. bioRxiv 2026.04.25.720787. https://doi.org/10.64898/2026.04.25.720787

## License

PolyForm Noncommercial License 1.0.0 (see `LICENSE`). For commercial use contact the authors.
