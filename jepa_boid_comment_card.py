"""One picture for the LinkedIn comment: three levels kept apart (latent dimension, BOID, autonomous regime) and the two traces along a run.
Reads medium_SG_poles.json, medium_SG_checks.json, medium_SG_figure_counts.json; writes jepa_boid_comment_card.png (1600x900)."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "medium_SG_poles.json"))); C = json.load(open(os.path.join(HERE, "medium_SG_checks.json"))); F = json.load(open(os.path.join(HERE, "medium_SG_figure_counts.json")))
R, CR, RO = P["runs"], C["runs"], P["rollouts"]
tr_dim = [r["d_hankel"] for r in R]; tr_boid = [c["above_AR1_resolved_pairs"] for c in CR]
un = [u for row in C["untrained"] for u in row]; un_dim = [u["d_hankel"] for u in un]; un_boid = [u["above_AR1_pairs"] for u in un]
px_dim = [x["d_hankel"] for x in C["input"]]; px_boid = [x["above_AR1_pairs"] for x in C["input"]]
ro_dim = [r["d_hankel"] for r in RO] + [int(x) for x in F["rollout_dim"]]; ro_boid = [r["pairs_resolved"] for r in RO] + [int(x) for x in F["rollout_boid"]]
BLUE, GREEN, PINK, GREY = "#1f77b4", "#2ca02c", "#d81b60", "0.45"
fig = plt.figure(figsize=(16, 9), dpi=100); fig.patch.set_facecolor("white")
fig.text(0.5, 0.955, "A JEPA latent read in time (toy, stop-gradient):\nrich in dimension, poor in rhythms, no autonomous regime under an open-loop rollout", ha="center", va="center", fontsize=16, fontweight="bold", color="0.15", linespacing=1.3)
fig.text(0.5, 0.893, "three numbers kept apart, because a count of latent components is not a count of oscillators  ·  three held-out runs of 1024 frames, lag 16", ha="center", va="center", fontsize=12.5, color="0.35")
# ---- left: the three levels as range bars
ax = fig.add_axes([0.20, 0.14, 0.34, 0.64])
rows = [("the frames themselves\n(32 principal comps)", px_dim, px_boid, GREY),
        ("untrained encoder\n(3 initialisations)", un_dim, un_boid, "0.6"),
        ("trained encoder,\nwith input", tr_dim, tr_boid, BLUE),
        ("predictor on its own\n(300-step rollout)", ro_dim, ro_boid, PINK)]
y = np.arange(len(rows))[::-1]*2.0
for yi, (lab, dim, boid, col) in zip(y, rows):
    ax.barh(yi + 0.42, max(dim) - min(dim) + 0.6, left=min(dim) - 0.3, height=0.7, color=col, alpha=0.85)
    ax.barh(yi - 0.42, max(boid) - min(boid) + 0.6, left=min(boid) - 0.3, height=0.7, color=GREEN, alpha=0.85)
    ax.text(max(dim) + 1.2, yi + 0.42, f"latent dimension {min(dim)}–{max(dim)}" if min(dim) != max(dim) else f"latent dimension {min(dim)}", va="center", fontsize=11, color="0.2")
    ax.text(max(boid) + 1.2, yi - 0.42, f"BOID {min(boid)}–{max(boid)}" if min(boid) != max(boid) else f"BOID {min(boid)}", va="center", fontsize=11, color=GREEN, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11); ax.set_xlim(0, 100); ax.set_xlabel("count", fontsize=11)
ax.set_title("latent dimension (Hankel components above a permutation null)\nversus BOID (ESPRIT pairs above a red-noise null, two cycles observed)", fontsize=11, loc="left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(axis="x", alpha=0.25)
# ---- right: along run 0
ax2 = fig.add_axes([0.62, 0.14, 0.355, 0.64]); N = 1024
rc, rd, rb = np.array(F["real_centre"]), np.array(F["real_dim"]), np.array(F["real_boid"]); oc, od, ob = N + np.array(F["rollout_centre"]), np.array(F["rollout_dim"]), np.array(F["rollout_boid"])
ax2.plot(rc, rd, color=BLUE, lw=2.5, marker="o", ms=5, label="latent dimension, with input")
ax2.plot(rc, rb, color=GREEN, lw=2.5, marker="s", ms=5, label="BOID, with input")
ax2.plot(oc, od, color=PINK, lw=2.5, marker="o", ms=5, label="latent dimension, predictor on its own")
ax2.plot(oc, ob, color=PINK, lw=1.6, ls="--", marker="s", ms=5, label="BOID, predictor on its own")
ax2.axvline(N, color="0.5", ls=":", lw=1.2); ax2.text(N - 12, rd.max() + 1.0, "rollout starts\n(open loop, zero actions)", fontsize=9.5, color="0.4", ha="right")
ax2.set_ylim(0, rd.max() + 6); ax2.set_xlabel("frame", fontsize=11); ax2.set_ylabel("count in a 256-frame window", fontsize=11); ax2.legend(fontsize=9.5, loc="upper left", ncol=2, frameon=False)
ax2.set_title(f"one held-out run: dimension follows the agent's freedom to move ({rd.min()}–{rd.max()});\nBOID stays at {rb.min()}–{rb.max()}; alone: {int(np.median(od))} components at the threshold, no oscillation", fontsize=11, loc="left")
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.grid(alpha=0.25)
fig.text(0.5, 0.055, "One seed, one toy; the real question is what SVReg and Slow SVReg do at these levels on the released model.", ha="center", fontsize=11, color="0.35")
fig.text(0.5, 0.025, "Bernadotte & Menshikov, Aicumene 2026  ·  github.com/aicumene/BOID-JEPA  ·  research.aicumene.com/papers/BOID-JEPA/", ha="center", fontsize=11, color="0.35")
fig.savefig(os.path.join(HERE, "jepa_boid_comment_card.png"), dpi=100, facecolor="white"); print("card saved")
