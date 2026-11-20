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

Measured on held-out trajectories: the BOID count (components above the 95 % permutation-surrogate threshold), the poles of the leading modes, the share of latent power at periods under five frames (jitter), the jaggedness of the latent path, the linear readout R² of the agent and block positions, and the count on a 400-step autonomous rollout of the predictor with the actions held at zero.

## First numbers (stop-gradient, maze2d-medium layout, three held-out runs of 1024 frames)

| quantity | run 0 | run 1 | run 2 |
|---|---|---|---|
| components above the surrogate threshold (BOID count) | 23 | 22 | 20 |
| count in 256-frame windows along the run | 11–19 | 10–20 | 10–18 |
| correlation of the windowed count with the agent's mean speed | +0.69 | +0.50 | +0.36 |
| correlation with the share of frames stuck at a wall | −0.71 | −0.44 | −0.72 |
| components on a 300-step autonomous rollout | 2 | 2 | 2 |
| jitter (share of power at periods < 5 frames) | 0.079 | 0.044 | 0.046 |
| jaggedness of the latent path (1 = smooth; Ren et al. report 5.3 for SG) | 4.15 | 4.22 | 3.82 |
| readout R², agent position (ridge, 5-fold CV) | 0.16 | 0.62 | 0.58 |
| readout R², block position | 0.51 | not pushed | 0.74 |
| leading periods, frames | 123, 55, 19, 15 | 1184, 409, 238, 31 | 1198, 274, 68, 28 |

**When the regime changes.** The windowed count follows how freely the agent moves, in all three runs: high where it travels unobstructed, low where it is pressed against walls (windows with a stuck share of 0.23–0.27 give 10–12 components, windows with 0.13–0.17 give 17–20). Rooms versus corridors do not decide it (the correlation with the share of frames in open cells changes sign between runs), nor does contact with the block. The regimes of this latent are regimes of the input; the predictor on its own has none (`medium_SG_regimes.json`).

`jepa_boid_latent_3d.png` and `jepa_boid_latent_3d.mp4`: the latent in PC1–PC3 coloured by the agent's position, the maze with the same colours, and the BOID count along the run and into the rollout. The readout is a cross-validated ridge regression on the standardised latent (a split-halves readout fails on these runs because the agent visits different rooms in the two halves). An earlier U-maze version of the toy (`run_umaze/` in the working folder) gave 16–19 components, jaggedness 4.4 and the same collapse to 2 on the rollout.

We stopped at the stop-gradient model on purpose: the two regularized regimes should be measured on the released model, where the question is real, rather than on a toy.

## Run

```
pip install numpy torch matplotlib
python jepa_boid_toy.py 3000 3      # steps, seeds; writes jepa_boid_results.json, latents_*.npz, model_*.pt
python jepa_boid_figure.py          # latent maps, counts, spectra, readouts
python jepa_boid_3d.py              # 3-D latent map with the BOID trace, PNG and rotating MP4
```

## References

- Ren M., Wang Y., Hoang C., Kuang Y., LeCun Y. (2026). The content–geometry tradeoff in world models: when JEPA learns latent maps. AGL-TR-2026-04. https://doi.org/10.5281/zenodo.24356797 · alphaXiv: https://www.alphaxiv.org/abs/2611.jepa-latent-maps-content-geometry · project page: https://agenticlearning.ai/content-geometry-tradeoff/
- Bernadotte A., Menshikov I. (2026). From Brain Dynamics to Strong Artificial Intelligence: Learning Creates Functional Regime Switching. Aicumene Preprint 2026.045v2. https://doi.org/10.13140/RG.2.2.22357.59364 · https://research.aicumene.com/papers/BOID-JEPA/
- Bernadotte A., Menshikov I. (2026). Multichannel Singular Spectrum Analysis: Counting Oscillatory Modes and Recovering Their Spatial Loadings. SSRN 7693202. https://doi.org/10.2139/ssrn.7693202 · code https://github.com/aicumene/MSSA
- Bernadotte A., Menshikov I. (2026). Brain Oscillator Intrinsic Dimension Marks Consciousness, Recovery Mode, and Failed Downshifting in Coma. SSRN 7667278. https://doi.org/10.2139/ssrn.7667278
- Bernadotte A. (2026). Topology-driven classification of time series. bioRxiv 2026.04.25.720787. https://doi.org/10.64898/2026.04.25.720787

## License

PolyForm Noncommercial License 1.0.0 (see `LICENSE`). For commercial use contact the authors.
