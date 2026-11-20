"""AR(1) red-noise null for the untrained encoder and the pixel input, and the windowed d_Hankel of the three runs with 200 surrogates."""
import numpy as np, torch, json, os, sys
sys.argv = [sys.argv[0]]; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jepa_boid_toy as J
from jepa_boid_poles import gram, std, d_hankel, esprit, classify, ar1_surrogate, L0
rng = np.random.default_rng(2); z = np.load(os.path.join(HERE, "latents_SG_seed0.npz"))
held = [J.rollout(1024, np.random.default_rng(100+k)) for k in range(3)]; frames = [J.render(P, Q) for (A, P, Q) in held]
def above_ar1(Z, B=60):
    lam = gram(Z, L0)[0]; sur = np.array([gram(ar1_surrogate(Z, rng), L0)[0] for _ in range(B)]); c = int(np.sum(lam > np.quantile(sur, 0.95, axis=0)))
    cl = classify(esprit(Z, L0, max(c, 2)), Z.shape[1]) if c >= 2 else dict(resolved=[]); return c, len(cl["resolved"]), cl["resolved"]
out = json.load(open(os.path.join(HERE, "medium_SG_checks.json")))
for seed in range(3):
    torch.manual_seed(seed); e0 = J.Encoder(); e0.eval()
    for k in range(3):
        with torch.no_grad(): Z0 = e0(torch.tensor(frames[k])[:, None]).numpy().T
        c, p, per = above_ar1(std(Z0)); out["untrained"][seed][k].update(above_AR1=c, above_AR1_pairs=p, above_AR1_periods=per)
    print(f"untrained seed {seed}: above AR(1) " + "; ".join(f"run {k}: {out['untrained'][seed][k]['above_AR1']} comps, {out['untrained'][seed][k]['above_AR1_pairs']} pairs {out['untrained'][seed][k]['above_AR1_periods']}" for k in range(3)))
for k in range(3):
    Xf = frames[k].reshape(1024, -1); Xf = Xf - Xf.mean(0); U, S, Vt = np.linalg.svd(Xf, full_matrices=False); Zp = (Xf @ Vt[:32].T).T
    c, p, per = above_ar1(std(Zp)); out["input"][k].update(above_AR1=c, above_AR1_pairs=p, above_AR1_periods=per); print(f"input PCA-32 run {k}: above AR(1) {c} comps, {p} pairs {per[:8]}")
out["windowed_B200"] = []
for k in range(3):
    Z = z[f"Z{k}"]; wc = [d_hankel(std(Z[:, s:s+256]), L0, rng, B=200)[0] for s in range(0, 1024-256+1, 32)]; out["windowed_B200"].append(wc); print(f"run {k} windowed d_Hankel (B=200): {wc}")
json.dump(out, open(os.path.join(HERE, "medium_SG_checks.json"), "w"), indent=1)
