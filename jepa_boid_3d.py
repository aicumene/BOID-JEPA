"""3-D latent map of the toy stop-gradient JEPA in the style of Ren et al. (white background, PC1–PC3, colour = position),
joined with the BOID regimes: the sliding-window count along the run and into the autonomous rollout.
Writes jepa_boid_latent_3d.png and the rotating clip jepa_boid_latent_3d.mp4.  Usage: python jepa_boid_3d.py [latents.npz]"""
import numpy as np, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "latents_SG_seed0.npz"); sys.argv = [sys.argv[0]]
import jepa_boid_toy as J
d = np.load(path); Z, P, Zi = d["Z0"], d["P0"], d["Zimag0"][:, :300]          # rollout truncated to 300 steps (it leaves the data region later)
mu, sd = Z.mean(1, keepdims=True), Z.std(1, keepdims=True) + 1e-8; Zs = (Z - mu)/sd; Zis = (Zi - mu)/sd
U, S, Vt = np.linalg.svd(Zs, full_matrices=False); W = U[:, :3]; pc = W.T @ Zs; pci = W.T @ Zis; ev = S**2/(S**2).sum()
WIN, STEP = 256, 32; rng = np.random.default_rng(0)
def sliding(Zx, win=WIN, step=STEP):
    return np.array([(s + win//2, J.boid_count(Zx[:, s:s+win], L=16, Bsur=30, rng=rng)[0]) for s in range(0, Zx.shape[1] - win + 1, step)])
cnt_real = sliding(Zs); cnt_roll = sliding(Zis, win=200, step=50)
print("real run counts:", cnt_real[:, 1].tolist()); print("rollout counts:", cnt_roll[:, 1].tolist())
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
    axc.plot(cnt_real[:, 0], cnt_real[:, 1], color="C0", lw=2.2, marker="o", ms=4, label="with input: held-out trajectory")
    axc.plot(Z.shape[1] + cnt_roll[:, 0], cnt_roll[:, 1], color=PINK, lw=2.2, marker="o", ms=4, label="no input: predictor on its own")
    axc.axvline(Z.shape[1], color="0.5", ls=":", lw=1); axc.text(Z.shape[1] + 10, cnt_real[:, 1].max() + 0.6, "rollout starts", fontsize=8, color="0.4")
    axc.set_ylim(0, cnt_real[:, 1].max() + 3); axc.set_xlabel("frame"); axc.set_ylabel("BOID: components above threshold"); axc.legend(fontsize=8, loc="upper left")
    axc.set_title("how BOID changes along the run (256-frame windows)", fontsize=10); axc.grid(alpha=0.3)
    fig.text(0.64, 0.045, f"with input the latent sustains {int(np.median(cnt_real[:, 1]))} components (median; range {int(cnt_real[:, 1].min())}–{int(cnt_real[:, 1].max())});\nwithout input it falls to {int(np.median(cnt_roll[:, 1]))}.   github.com/aicumene/BOID-JEPA", fontsize=8.5, color="0.3", va="center")
fig = plt.figure(figsize=(12.8, 7.2), dpi=150); draw(fig, azim=-55); fig.savefig(os.path.join(HERE, "jepa_boid_latent_3d.png"), facecolor="white"); plt.close(fig); print("png saved")
import imageio.v2 as imageio
wr = imageio.get_writer(os.path.join(HERE, "jepa_boid_latent_3d.mp4"), fps=30, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None); n = 360
for i in range(n):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100); draw(fig, azim=-55 + 360*i/n, elev=24 + 6*np.sin(2*np.pi*i/n)); fig.canvas.draw()
    wr.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
wr.close(); print("mp4 saved")
