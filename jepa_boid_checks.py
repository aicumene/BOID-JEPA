"""Controls for Section 7.9 (stop-gradient toy JEPA, maze2d-medium), on the saved model and latents.
A  components above an aperiodic null with the same statistic as d_Hankel (sorted Gram eigenvalues against AR(1) and 1/f^chi surrogates);
B  spectral peaks: mean standardized Welch spectrum of the latent against the 95 % band of 1/f^chi surrogates;
C  the autonomous rollout: d_Hankel with 200 surrogates, where the fixed point lies relative to the data cloud, and an open-loop
   rollout driven by the real action sequence (own latent history, real actions) to separate the zero-action setting from the
   predictor's own dynamics;
D  untrained control: the same encoder at random initialisation (three seeds) on the same held-out frames;
E  input control: the same count on the first 32 principal components of the frames themselves, whole-run and in the 256-frame windows."""
import numpy as np, torch, json, os, sys
sys.argv = [sys.argv[0]]; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jepa_boid_toy as J
from jepa_boid_poles import block_hankel, gram, std, d_hankel, esprit, classify, ar1_surrogate, powerlaw_surrogate, L0
rng = np.random.default_rng(1); z = np.load(os.path.join(HERE, "latents_SG_seed0.npz")); out = {}
held = [J.rollout(1024, np.random.default_rng(100+k)) for k in range(3)]
assert all(np.allclose(P, z[f"P{k}"]) for k, (A, P, Q) in enumerate(held)), "held-out runs regenerated differently"
frames = [J.render(P, Q) for (A, P, Q) in held]
def d_coloured(Z, L, null, B=60):
    lam = gram(Z, L)[0]; sur = np.array([gram(powerlaw_surrogate(Z, rng) if null == "1/f" else ar1_surrogate(Z, rng), L)[0] for _ in range(B)])
    return int(np.sum(lam > np.quantile(sur, 0.95, axis=0)))
def welch(Z, seg=256):
    w = np.hanning(seg); f = np.fft.rfftfreq(seg); S = np.zeros(len(f))
    for x in Z:
        for s in range(0, Z.shape[1]-seg+1, seg//2): S += np.abs(np.fft.rfft((x[s:s+seg]-x[s:s+seg].mean())*w))**2
    return f, S/Z.shape[0]
def spectral_peaks(Z, B=100):
    f, S = welch(Z); sur = np.array([welch(powerlaw_surrogate(Z, rng))[1] for _ in range(B)]); q = np.quantile(sur, 0.95, axis=0)
    above = S > q; peaks = [i for i in range(1, len(f)-1) if above[i] and S[i] >= S[i-1] and S[i] >= S[i+1]]
    return [round(1/f[i]) for i in peaks], int(above[1:].sum())
# ---- A, B on the held-out runs
out["runs"] = []
for k in range(3):
    Zs = std(z[f"Z{k}"]); a = dict(above_AR1=d_coloured(Zs, L0, "AR1"), above_1f=d_coloured(Zs, L0, "1/f"))
    for key in ("above_AR1", "above_1f"):
        c = a[key]; cl = classify(esprit(Zs, L0, max(c, 2)), 1024) if c >= 2 else dict(resolved=[], real=[])
        a[key + "_resolved_pairs"] = len(cl["resolved"]); a[key + "_periods"] = cl["resolved"]
    a["spectral_peaks_periods"], a["spectral_bins_above"] = spectral_peaks(Zs); out["runs"].append(a)
    print(f"run {k}: above AR(1) {a['above_AR1']} ({a['above_AR1_resolved_pairs']} resolved pairs {a['above_AR1_periods']}); above 1/f^chi {a['above_1f']} ({a['above_1f_resolved_pairs']} pairs {a['above_1f_periods']}); spectral peaks above the 1/f band at periods {a['spectral_peaks_periods']} ({a['spectral_bins_above']} bins)")
# ---- C rollout
sd = torch.load(os.path.join(HERE, "model_SG_seed0.pt")); enc, pred = J.Encoder(), J.Predictor(gated=False); enc.load_state_dict(sd["enc"]); pred.load_state_dict(sd["pred"]); enc.eval(); pred.eval()
@torch.no_grad()
def open_loop(X, A, steps=300):
    zz = enc(torch.tensor(X[:3])[:, None]); hist = [zz[0], zz[1], zz[2]]; o = []
    for t in range(steps):
        zn = pred(torch.stack(hist[-3:], 0)[None], torch.tensor(A[3+t:4+t]))[0]; hist.append(zn); o.append(zn.numpy())
    return np.array(o).T
@torch.no_grad()
def one_step(X, A):
    """teacher-forced: real history and real action at every step"""
    zz = enc(torch.tensor(X)[:, None]); z3 = torch.stack([zz[:-3], zz[1:-2], zz[2:-1]], 1); return pred(z3, torch.tensor(A[2:-1])).numpy().T, zz.numpy().T
out["rollouts"] = []
for k in range(3):
    A, P, Q = held[k]; X = frames[k]; Zreal = z[f"Z{k}"]; Zi = z[f"Zimag{k}"][:, :300]
    d0, _ = d_hankel(std(Zi), L0, rng, B=200); cl0 = classify(esprit(std(Zi), L0, max(d0, 2)), 300)
    wins = [d_hankel(std(Zi[:, s:s+256]), L0, rng, B=200)[0] for s in range(0, 300-256+1, 32)]
    # where the fixed point lies: nearest data latent, in units of the data's own nearest-neighbour distance and of its radius
    nn_data = np.median(np.sort(np.linalg.norm(Zreal.T[:, None] - Zreal.T[None], axis=-1), axis=1)[:, 1]); radius = np.linalg.norm(Zreal - Zreal.mean(1, keepdims=True), axis=0).mean()
    fp = Zi[:, -1]; d_fp = np.linalg.norm(Zreal.T - fp, axis=1).min(); d_c = np.linalg.norm(fp - Zreal.mean(1))
    Zo = open_loop(X, A); do_, _ = d_hankel(std(Zo), L0, rng, B=200); clo = classify(esprit(std(Zo), L0, max(do_, 2)), 300)
    step_o = np.linalg.norm(np.diff(Zo, axis=1), axis=0); d_o_fp = np.linalg.norm(Zreal.T - Zo[:, -1], axis=1).min()
    Zt, Zenc = one_step(X, A); dt_, _ = d_hankel(std(Zt), L0, rng, B=60); err = float(np.mean((Zt - Zenc[:, 3:])**2)/np.var(Zenc))
    r = dict(d_hankel_B200=d0, poles=dict(resolved=cl0["resolved"], unresolved=cl0["unresolved"], real=cl0["real"]), windowed_B200=wins,
             fixed_point_dist_nearest_data=float(d_fp), data_nn_median=float(nn_data), fixed_point_dist_centre=float(d_c), data_radius=float(radius),
             open_loop_real_actions=dict(d_hankel=do_, resolved=clo["resolved"], unresolved=clo["unresolved"], real=clo["real"], step_first50=float(step_o[:50].mean()), step_last50=float(step_o[-50:].mean()), dist_nearest_data_final=float(d_o_fp)),
             teacher_forced=dict(d_hankel=dt_, rel_mse=err))
    out["rollouts"].append(r)
    print(f"rollout {k}: zero actions d={d0} (B=200) poles {r['poles']} windows {wins}; fixed point: {d_fp:.2f} from nearest data (data NN median {nn_data:.2f}, radius {radius:.2f}), {d_c:.2f} from centre")
    print(f"   real actions, own history: d={do_} resolved {clo['resolved']} unresolved {clo['unresolved']} real {clo['real']}; step {step_o[:50].mean():.4f} -> {step_o[-50:].mean():.4f}; final point {d_o_fp:.2f} from nearest data")
    print(f"   teacher-forced one-step predictions: d={dt_}, relative mse {err:.3f}")
# ---- D untrained control
out["untrained"] = []
for seed in range(3):
    torch.manual_seed(seed); e0 = J.Encoder(); e0.eval(); row = []
    for k in range(3):
        with torch.no_grad(): Z0 = e0(torch.tensor(frames[k])[:, None]).numpy().T
        Zs = std(Z0); d, _ = d_hankel(Zs, L0, rng); cl = classify(esprit(Zs, L0, max(d, 2)), 1024); c1f = d_coloured(Zs, L0, "1/f", B=40)
        row.append(dict(d_hankel=d, resolved=len(cl["resolved"]), unresolved=len(cl["unresolved"]), real=len(cl["real"]), above_1f=c1f, periods=cl["resolved"]))
    out["untrained"].append(row); print(f"untrained seed {seed}: " + "; ".join(f"run {k}: d={r['d_hankel']} pairs {r['resolved']}+{r['unresolved']}u real {r['real']} above1/f {r['above_1f']}" for k, r in enumerate(row)))
# ---- E input control: PCA-32 of the frames
out["input"] = []; reg = json.load(open(os.path.join(HERE, "medium_SG_regimes.json")))
for k in range(3):
    Xf = frames[k].reshape(1024, -1); Xf = Xf - Xf.mean(0); U, S, Vt = np.linalg.svd(Xf, full_matrices=False); Zp = (Xf @ Vt[:32].T).T
    Zs = std(Zp); d, _ = d_hankel(Zs, L0, rng); cl = classify(esprit(Zs, L0, max(d, 2)), 1024); c1f = d_coloured(Zs, L0, "1/f", B=40)
    wc = [d_hankel(std(Zp[:, s:s+256]), L0, rng)[0] for s in range(0, 1024-256+1, 32)]; lat = [w["count"] for w in reg[str(k)]]
    rho = float(np.corrcoef(wc, lat)[0, 1]); var32 = float((S[:32]**2).sum()/(S**2).sum())
    out["input"].append(dict(d_hankel=d, resolved=len(cl["resolved"]), unresolved=len(cl["unresolved"]), real=len(cl["real"]), above_1f=c1f, windowed=wc, corr_with_latent_windowed=rho, var_share_32pc=var32))
    print(f"input PCA-32 run {k}: d={d} pairs {len(cl['resolved'])}+{len(cl['unresolved'])}u real {len(cl['real'])} above1/f {c1f}; windowed {min(wc)}-{max(wc)}; corr with latent windowed count {rho:+.2f}; 32 PCs carry {var32:.2f} of pixel variance")
json.dump(out, open(os.path.join(HERE, "medium_SG_checks.json"), "w"), indent=1)
