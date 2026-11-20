"""Figure for Section 9.1: latent maps, BOID counts, spectra and readouts of the three toy JEPA objectives.
Reads latents_<variant>_seed<k>.npz and jepa_boid_results.json written by jepa_boid_toy.py."""
import numpy as np, json, glob, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); VARIANTS = ["SG", "SVReg", "SlowSVReg"]; LABELS = {"SG": "stop-gradient", "SVReg": "SVReg", "SlowSVReg": "Slow SVReg"}
WALLS = [(0.0, 0.0, 1.0, 0.06), (0.0, 0.94, 1.0, 0.06), (0.0, 0.0, 0.06, 1.0), (0.94, 0.0, 0.06, 1.0), (0.30, 0.35, 0.40, 0.30)]
def parse_log():
    """fall back to the per-run lines of the log while the run is in progress"""
    import re
    res = {}
    for line in open(os.path.join(HERE, "run_3000x3.log")):
        m = re.match(r"\s+(\w+) seed (\d+): count \[([\d, ]+)\] \| imagined \[([\d, ]+)\] \| jitter ([\d.]+) \| jag ([\d.]+) \| R2 agent (-?[\d.]+) block (-?[\d.]+)", line)
        if m:
            v = m.group(1); res.setdefault(v, []).append(dict(count=[int(x) for x in m.group(3).split(",")], count_imag=[int(x) for x in m.group(4).split(",")],
                jitter=[float(m.group(5))], jag=[float(m.group(6))], r2_agent=[float(m.group(7))], r2_block=[float(m.group(8))]))
    return res or None
res = json.load(open(os.path.join(HERE, "jepa_boid_results.json"))) if os.path.exists(os.path.join(HERE, "jepa_boid_results.json")) else parse_log()
VARIANTS = [v for v in VARIANTS if res and v in res] or VARIANTS
fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.6), dpi=160)
for k, v in enumerate(VARIANTS):
    files = sorted(glob.glob(os.path.join(HERE, f"latents_{v}_seed*.npz")))
    a = ax[0, k]
    if not files: a.set_title(f"{LABELS[v]}: not available yet"); a.axis("off"); continue
    d = np.load(files[0]); Z, P = d["Z0"], d["P0"]                              # held-out run 0 of seed 0
    Zc = Z - Z.mean(1, keepdims=True); U, S, Vt = np.linalg.svd(Zc, full_matrices=False); pc = (U[:, :2].T @ Zc)
    col = plt.cm.viridis((P[:, 0] - P[:, 0].min())/(np.ptp(P[:, 0]) + 1e-9))
    a.scatter(pc[0], pc[1], s=4, c=col, alpha=0.7, lw=0); a.plot(pc[0, :200], pc[1, :200], color="k", lw=0.5, alpha=0.6)
    a.set_title(f"{LABELS[v]}: latent PCA (PC1–PC2), colour = agent x", fontsize=9); a.set_xticks([]); a.set_yticks([])
    a.text(0.02, 0.97, f"explained {100*(S[:2]**2).sum()/(S**2).sum():.0f} %", transform=a.transAxes, va="top", fontsize=8)
    if k == 0:
        ins = a.inset_axes([0.68, 0.02, 0.30, 0.30]); ins.set_xlim(0, 1); ins.set_ylim(0, 1); ins.set_xticks([]); ins.set_yticks([])
        for (x, y, w, h) in WALLS: ins.add_patch(plt.Rectangle((x, y), w, h, color="0.4"))
        ins.scatter(P[::4, 0], P[::4, 1], s=2, c=col[::4], lw=0); ins.set_title("maze, agent path", fontsize=7)
# counts
a = ax[1, 0]
if res:
    for k, v in enumerate(VARIANTS):
        if v not in res: continue
        c = [np.mean(r["count"]) for r in res[v]]; ci = [np.mean(r["count_imag"]) for r in res[v]]
        a.bar(k - 0.18, np.mean(c), 0.34, color="C0", alpha=0.85, label="held-out trajectories" if k == 0 else None); a.scatter([k - 0.18]*len(c), c, color="k", s=12, zorder=3)
        a.bar(k + 0.18, np.mean(ci), 0.34, color="C3", alpha=0.85, label="autonomous rollout, 400 steps" if k == 0 else None); a.scatter([k + 0.18]*len(ci), ci, color="k", s=12, zorder=3)
    a.set_xticks(range(len(VARIANTS))); a.set_xticklabels([LABELS[v] for v in VARIANTS]); a.set_ylabel("components above the surrogate threshold"); a.legend(fontsize=7); a.set_title("BOID count of the latent (dots: seeds)", fontsize=9)
# spectra
a = ax[1, 1]
for k, v in enumerate(VARIANTS):
    files = sorted(glob.glob(os.path.join(HERE, f"latents_{v}_seed*.npz")))
    if not files: continue
    spec = []
    for f in files:
        d = np.load(f)
        for run in range(3):
            Z = d[f"Z{run}"]; Zc = Z - Z.mean(1, keepdims=True); Zc /= Zc.std() + 1e-9
            Pw = (np.abs(np.fft.rfft(Zc, axis=1))**2).sum(0); spec.append(Pw/Pw[1:].sum())
    fr = np.fft.rfftfreq(Z.shape[1]); m = np.mean(spec, 0)
    a.loglog(fr[1:], m[1:], lw=1.2, label=LABELS[v])
a.axvline(0.2, color="0.5", ls=":", lw=1); a.text(0.21, a.get_ylim()[1]*0.5 if a.get_ylim()[1] > 0 else 1, "period 5 steps", fontsize=7, color="0.4")
a.set_xlabel("frequency, cycles per frame"); a.set_ylabel("share of latent power"); a.set_title("power spectrum of the latent (all channels)", fontsize=9); a.legend(fontsize=7)
# readouts and geometry
a = ax[1, 2]
if res:
    keys = [("r2_agent", "R² agent position"), ("r2_block", "R² block position"), ("jitter", "jitter (power at periods < 5)"), ("jag", "jaggedness / 10")]
    for j, (key, lab) in enumerate(keys):
        vals = [np.mean([np.mean(r[key]) for r in res[v]]) if v in res else 0 for v in VARIANTS]; vals = [x/10 if key == "jag" else x for x in vals]
        a.bar(np.arange(len(VARIANTS)) + (j - 1.5)*0.2, np.clip(vals, -0.2, None), 0.2, label=lab)
    a.set_xticks(range(len(VARIANTS))); a.set_xticklabels([LABELS[v] for v in VARIANTS]); a.axhline(0, color="k", lw=0.5); a.legend(fontsize=7); a.set_title("content and geometry of the latent", fontsize=9)
fig.suptitle("BOID for JEPA, toy U-maze with a pushable block: the same encoder and predictor under three objectives", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96]); out = os.path.join(HERE, "jepa_boid_figure.png"); fig.savefig(out); print("saved", out)
