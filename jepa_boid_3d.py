"""3-D latent map of the toy stop-gradient JEPA in the style of Ren et al. (white background, PC1–PC3, colour = position),
joined with the regimes along the run: the latent (Hankel) dimension and BOID proper, the oscillators above a red-noise null, in sliding windows,
and the same on the autonomous rollout.  Counts are written to medium_SG_figure_counts.json.
Writes jepa_boid_latent_3d.png and the rotating clip jepa_boid_latent_3d.mp4.  Usage: python jepa_boid_3d.py [latents.npz]"""
import numpy as np, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "latents_SG_seed0.npz"); sys.argv = [sys.argv[0]]
import jepa_boid_toy as J, json
from jepa_boid_poles import gram, esprit, classify, ar1_surrogate, std as stdz
d = np.load(path); Z, P, Zi = d["Z0"], d["P0"], d["Zimag0"][:, :300]          # rollout truncated to 300 steps (it leaves the data region later)
mu, sd = Z.mean(1, keepdims=True), Z.std(1, keepdims=True) + 1e-8; Zs = (Z - mu)/sd; Zis = (Zi - mu)/sd
U, S, Vt = np.linalg.svd(Zs, full_matrices=False); W = U[:, :3]; pc = W.T @ Zs; pci = W.T @ Zis; ev = S**2/(S**2).sum()
WIN, STEP = 256, 32; rng = np.random.default_rng(0)
def boid_window(Zw, B=60):
    """BOID proper: resolved ESPRIT pairs (|z| >= 0.95, period <= half the window) among the components above an AR(1) red-noise null"""
    Zw = stdz(Zw); lam = gram(Zw, 16)[0]; sur = np.array([gram(ar1_surrogate(Zw, rng), 16)[0] for _ in range(B)]); c = int(np.sum(lam > np.quantile(sur, 0.95, axis=0)))
    return len(classify(esprit(Zw, 16, max(c, 2)), Zw.shape[1])["resolved"]) if c >= 2 else 0
def sliding(Zx, win=WIN, step=STEP):
    return np.array([(s + win//2, J.boid_count(Zx[:, s:s+win], L=16, Bsur=200, rng=rng)[0], boid_window(Zx[:, s:s+win])) for s in range(0, Zx.shape[1] - win + 1, step)])
_cache = os.path.join(HERE, "medium_SG_figure_counts.json")
if os.path.exists(_cache):
    _c = json.load(open(_cache)); cnt_real = np.array([_c["real_centre"], _c["real_dim"], _c["real_boid"]]).T; cnt_roll = np.array([_c["rollout_centre"], _c["rollout_dim"], _c["rollout_boid"]]).T; print("counts loaded from cache")
else:
    cnt_real = sliding(Zs); cnt_roll = sliding(Zis)
print("real run: latent dimension", cnt_real[:, 1].tolist(), "BOID", cnt_real[:, 2].tolist()); print("rollout: latent dimension", cnt_roll[:, 1].tolist(), "BOID", cnt_roll[:, 2].tolist())
json.dump(dict(window=WIN, step=STEP, real_centre=cnt_real[:, 0].tolist(), real_dim=cnt_real[:, 1].tolist(), real_boid=cnt_real[:, 2].tolist(),
               rollout_window=WIN, rollout_step=STEP, rollout_centre=cnt_roll[:, 0].tolist(), rollout_dim=cnt_roll[:, 1].tolist(), rollout_boid=cnt_roll[:, 2].tolist()),
          open(os.path.join(HERE, "medium_SG_figure_counts.json"), "w"), indent=1)
# colour = position: hue by x, lightness by y (a 2-D colour map, as in their top-down panel)
def pos_colours(P):
    hx = (P[:, 0] - P[:, 0].min())/(np.ptp(P[:, 0]) + 1e-9); ly = (P[:, 1] - P[:, 1].min())/(np.ptp(P[:, 1]) + 1e-9)
    base = plt.cm.hsv(hx*0.85)[:, :3]; return np.clip(base*(0.45 + 0.55*ly[:, None]) + (1 - ly[:, None])*0.25, 0, 1)
col = pos_colours(P); PINK = "#d81b60"
def draw(fig, azim, elev=24):
    ax = fig.add_axes([0.0, 0.03, 0.56, 0.94], projection="3d"); ax.view_init(elev=elev, azim=azim)
    ax.scatter(pc[0], pc[1], pc[2], s=6, c=col, alpha=0.9, lw=0, depthshade=False); ax.plot(pc[0], pc[1], pc[2], color="0.25", lw=0.3, alpha=0.4)
    ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}%)", labelpad=-6); ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}%)", labelpad=-6); ax.set_zlabel(f"PC3 ({100*ev[2]:.0f}%)", labelpad=-6); ax.tick_params(labelsize=6, pad=-3)
    ax.set_title("stop-gradient latent of a held-out run (PC1–PC3), colour = agent position", fontsize=10)
    axr = fig.add_axes([0.01, 0.02, 0.20, 0.26], projection="3d"); axr.patch.set_alpha(0); axr.view_init(elev=elev, azim=azim)
    axr.plot(pci[0], pci[1], pci[2], color=PINK, lw=1.6); axr.scatter([pci[0, 0]], [pci[1, 0]], [pci[2, 0]], s=35, color="k", depthshade=False)
    axr.set_xticks([]); axr.set_yticks([]); axr.set_zticks([]); axr.set_title("predictor on its own,\n300 steps (own scale)", fontsize=8, color=PINK, pad=-6)
    ins = fig.add_axes([0.60, 0.56, 0.38, 0.40]); ins.set_xlim(0, 1); ins.set_ylim(0, 1); ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")
    for (x, y, w, h) in J.WALLS: ins.add_patch(plt.Rectangle((x, y), w, h, color="0.25"))
    ins.scatter(P[:, 0], P[:, 1], s=5, c=col, lw=0); ins.set_title("top-down maze, agent path, colour = position", fontsize=9)
    axc = fig.add_axes([0.64, 0.15, 0.34, 0.35])
    axc.plot(cnt_real[:, 0], cnt_real[:, 1], color="C0", lw=2.2, marker="o", ms=4, label="latent dimension, with input")
    axc.plot(cnt_real[:, 0], cnt_real[:, 2], color="C2", lw=2.2, marker="s", ms=4, label="BOID (rhythms above red noise), with input")
    axc.plot(Z.shape[1] + cnt_roll[:, 0], cnt_roll[:, 1], color=PINK, lw=2.2, marker="o", ms=4, label="latent dimension, predictor on its own")
    axc.plot(Z.shape[1] + cnt_roll[:, 0], cnt_roll[:, 2], color=PINK, lw=1.4, ls="--", marker="s", ms=4, label="BOID, predictor on its own")
    axc.axvline(Z.shape[1], color="0.5", ls=":", lw=1); axc.text(Z.shape[1] + 10, cnt_real[:, 1].max() + 0.6, "rollout starts", fontsize=8, color="0.4")
    axc.set_ylim(0, cnt_real[:, 1].max() + 9); axc.set_xlabel("frame"); axc.set_ylabel("components / oscillators"); axc.legend(fontsize=7, loc="upper left", ncol=1, frameon=False)
    axc.set_title("latent dimension and BOID along the run (256-frame windows)", fontsize=10); axc.grid(alpha=0.3)
    fig.text(0.62, 0.045, f"with input: {int(cnt_real[:, 1].min())}–{int(cnt_real[:, 1].max())} Hankel components, only {int(cnt_real[:, 2].min())}–{int(cnt_real[:, 2].max())} rhythms above red noise;\non its own: {int(np.median(cnt_roll[:, 1]))} components at the threshold, real poles, no oscillation (BOID {int(np.median(cnt_roll[:, 2]))}).\ngithub.com/aicumene/BOID-JEPA", fontsize=8, color="0.3", va="center")
fig = plt.figure(figsize=(12.8, 7.2), dpi=150); draw(fig, azim=-55); fig.savefig(os.path.join(HERE, "jepa_boid_latent_3d.png"), facecolor="white"); plt.close(fig); print("png saved")
import imageio.v2 as imageio
wr = imageio.get_writer(os.path.join(HERE, "jepa_boid_latent_3d.mp4"), fps=30, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None); n = 360
for i in range(n):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100); draw(fig, azim=-55 + 360*i/n, elev=24 + 6*np.sin(2*np.pi*i/n)); fig.canvas.draw()
    wr.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
wr.close(); print("mp4 saved")
