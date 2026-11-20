"""Three numbers instead of one, and their checks, for the toy JEPA latents (stop-gradient, maze2d-medium).

For each held-out run and for the autonomous rollout:
  d_Hankel   components of the block-Hankel Gram above the 95 % permutation-surrogate threshold (what the paper calls the count);
  pairs      positive-frequency TLS-ESPRIT poles at rank d_Hankel with |z| >= 0.95, split into resolved (period <= N/2, two cycles
             observed) and unresolved (period > N/2, indistinguishable from a trend); real poles are listed separately;
  above 1/f  Monte-Carlo SSA (Allen & Smith 1996): data eigenvalues above the 95 % quantile of surrogate variances projected on the
             data eigendirections, with two aperiodic nulls, AR(1) red noise and a fitted 1/f^chi spectrum; the oscillators among
             those components are counted by ESPRIT on that subspace (resolved pairs only).
Checks: stability of the resolved pairs across lag L in {8, 16, 32} and record length N in {512, 768, 1024}; a trend null, in which
each channel is detrended (linear trend and a 257-frame moving average removed) before the poles are refitted.
"""
import json, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
L0, ZMIN, B_PERM, B_MC = 16, 0.95, 40, 100

def block_hankel(Z, L):
    return np.vstack([np.lib.stride_tricks.sliding_window_view(Z[j], L).T for j in range(Z.shape[0])])
def gram(Z, L):
    """eigenvalues of the trajectory Gram matrix H H^T / K and its eigenvectors, by SVD, so that only the min(rows, columns) nonzero
    values are returned: on a record shorter than the embedding (K < mL) the trailing eigenvalues are numerical zeros and must not be
    compared with surrogates"""
    H = block_hankel(Z, L); U, sv, _ = np.linalg.svd(H, full_matrices=False); return sv**2 / H.shape[1], U
def std(Z): return (Z - Z.mean(1, keepdims=True))/(Z.std(1, keepdims=True) + 1e-8)
def d_hankel(Z, L, rng, B=B_PERM, eta=0.05):
    lam = gram(Z, L)[0]
    sur = np.array([gram(np.array([rng.permutation(x) for x in Z]), L)[0] for _ in range(B)]); q = np.quantile(sur, 1-eta, axis=0)
    return int(np.sum(lam > q)), lam
def esprit(Z, L, d):
    H = block_hankel(Z, L); U = np.linalg.svd(H, full_matrices=False)[0][:, :d]; m = Z.shape[0]; Bk = U.reshape(m, L, d)
    up, dn = Bk[:, :-1].reshape(m*(L-1), d), Bk[:, 1:].reshape(m*(L-1), d)
    return np.linalg.eigvals(np.linalg.lstsq(up, dn, rcond=None)[0])
def classify(z, N, zmin=ZMIN, tol=1e-3):
    ang, mod = np.angle(z), np.abs(z); pos = (ang > tol) & (ang < np.pi - tol); T = 2*np.pi/np.where(pos, np.abs(ang), 1)
    res = pos & (mod >= zmin) & (T <= N/2); unres = pos & (mod >= zmin) & (T > N/2); weak = pos & (mod < zmin)
    real = ~pos & (ang >= -tol); return dict(resolved=sorted(T[res].round(0).tolist(), reverse=True), unresolved=sorted(T[unres].round(0).tolist(), reverse=True),
                                             weak_pairs=int(weak.sum()), real=sorted(mod[real].round(3).tolist(), reverse=True), mod_res=mod[res].round(3).tolist())
# ---- aperiodic nulls for Monte-Carlo SSA
def ar1_surrogate(Z, rng):
    S = np.empty_like(Z)
    for j, x in enumerate(Z):
        g = np.corrcoef(x[:-1], x[1:])[0, 1]; g = min(max(g, -0.999), 0.999); e = rng.standard_normal(len(x)) * np.sqrt(1 - g*g)
        y = np.empty(len(x)); y[0] = rng.standard_normal()
        for t in range(1, len(x)): y[t] = g*y[t-1] + e[t]
        S[j] = (y - y.mean())/y.std()*x.std() + x.mean()
    return S
def powerlaw_surrogate(Z, rng):
    S = np.empty_like(Z); N = Z.shape[1]; f = np.fft.rfftfreq(N)
    for j, x in enumerate(Z):
        P = np.abs(np.fft.rfft(x - x.mean()))**2; k = f > 0; a, b = np.polyfit(np.log(f[k]), np.log(P[k] + 1e-12), 1)   # log P = a log f + b
        amp = np.sqrt(np.exp(b) * f[k]**a); ph = rng.uniform(0, 2*np.pi, k.sum()); spec = np.zeros(len(f), complex); spec[k] = amp*np.exp(1j*ph)
        y = np.fft.irfft(spec, n=N); S[j] = (y - y.mean())/y.std()*x.std() + x.mean()
    return S
def mcssa(Z, L, rng, null, B=B_MC, eta=0.05):
    lam, U = gram(Z, L); sur = []
    for _ in range(B):
        S = powerlaw_surrogate(Z, rng) if null == "1/f" else ar1_surrogate(Z, rng); Hs = block_hankel(S, L); sur.append(((U.T @ Hs)**2).mean(1))
    q = np.quantile(np.array(sur), 1-eta, axis=0); above = lam > q
    return int(above.sum())
def detrend(Z, k=257):
    out = np.empty_like(Z); t = np.arange(Z.shape[1])
    for j, x in enumerate(Z):
        x = x - np.polyval(np.polyfit(t, x, 1), t); pad = np.pad(x, k//2, mode="reflect"); ma = np.convolve(pad, np.ones(k)/k, mode="valid"); out[j] = x - ma
    return out
def matches(T, Tset, tol=0.2):
    return any(abs(T - T2) <= max(tol*T, 1.0) for T2 in Tset)

def analyse(Z, N, label, rng):
    Zs = std(Z[:, :N]); d, lam = d_hankel(Zs, L0, rng); cl = classify(esprit(Zs, L0, max(d, 2)), N)
    # oscillators above the aperiodic background: ESPRIT on the MC-SSA subspace
    out = dict(label=label, N=N, d_hankel=d, pairs_resolved=len(cl["resolved"]), pairs_unresolved=len(cl["unresolved"]), weak_pairs=cl["weak_pairs"],
               real_poles=cl["real"], periods_resolved=cl["resolved"], periods_unresolved=cl["unresolved"], mod_resolved=cl["mod_res"])
    for null in ("AR1", "1/f"):
        c = mcssa(Zs, L0, rng, null); cc = classify(esprit(Zs, L0, max(c, 2)), N) if c >= 2 else dict(resolved=[], unresolved=[])
        out[f"above_{null}"] = c; out[f"osc_above_{null}"] = len(cc["resolved"]); out[f"periods_above_{null}"] = cc["resolved"]
    # trend null: detrend, refit
    Zd = std(detrend(Z[:, :N])); dd, _ = d_hankel(Zd, L0, rng); cd = classify(esprit(Zd, L0, max(dd, 2)), N)
    out["detrended"] = dict(d_hankel=dd, pairs_resolved=len(cd["resolved"]), pairs_unresolved=len(cd["unresolved"]), real_poles=cd["real"], periods_resolved=cd["resolved"])
    out["resolved_surviving_detrend"] = int(sum(matches(T, cd["resolved"]) for T in cl["resolved"]))
    return out, cl

def stability(Z, rng, ref):
    """resolved pairs of the reference setting (L=16, N=1024) found again, within 20 % of the period, at |z|>=0.95 in every other setting"""
    sets = {}
    for L in (8, 16, 32):
        for N in (512, 768, 1024):
            Zs = std(Z[:, :N]); d, _ = d_hankel(Zs, L, rng); sets[(L, N)] = classify(esprit(Zs, L, max(d, 2)), N)["resolved"] + classify(esprit(Zs, L, max(d, 2)), N)["unresolved"]
    stable = [T for T in ref if all(matches(T, s) for s in sets.values())]
    across_L = [T for T in ref if all(matches(T, sets[(L, 1024)]) for L in (8, 16, 32))]
    across_N = [T for T in ref if all(matches(T, sets[(16, N)]) for N in (512, 768, 1024))]
    return dict(stable_all=stable, stable_across_lag=across_L, stable_across_length=across_N, d_hankel_by_setting={f"L{L}_N{N}": None for (L, N) in sets})

if __name__ == "__main__":
    z = np.load(os.path.join(HERE, "latents_SG_seed0.npz")); rng = np.random.default_rng(0); res = {"runs": [], "rollouts": []}
    for k in range(3):
        out, cl = analyse(z[f"Z{k}"], 1024, f"run {k}", rng); st = stability(z[f"Z{k}"], rng, cl["resolved"]); out.update(st); res["runs"].append(out)
        print(f"run {k}: d_Hankel={out['d_hankel']}  pairs resolved={out['pairs_resolved']} (T<=512) unresolved={out['pairs_unresolved']} weak={out['weak_pairs']} real={out['real_poles']}")
        print(f"   resolved periods {out['periods_resolved']} |z| {out['mod_resolved']}; unresolved {out['periods_unresolved']}")
        print(f"   stable across lag {len(st['stable_across_lag'])}, across length {len(st['stable_across_length'])}, across both {len(st['stable_all'])}: {st['stable_all']}")
        print(f"   MC-SSA above AR(1): {out['above_AR1']} comps, {out['osc_above_AR1']} resolved oscillators {out['periods_above_AR1']}; above 1/f^chi: {out['above_1/f']} comps, {out['osc_above_1/f']} osc {out['periods_above_1/f']}")
        print(f"   detrended: d_Hankel={out['detrended']['d_hankel']} resolved={out['detrended']['pairs_resolved']} unresolved={out['detrended']['pairs_unresolved']} real={out['detrended']['real_poles']}; resolved pairs surviving detrend {out['resolved_surviving_detrend']}/{out['pairs_resolved']}")
    for k in range(3):
        Zi = z[f"Zimag{k}"][:, :300]; out, cl = analyse(Zi, 300, f"rollout {k}", rng)
        # windowed count as in the figure, and the poles of those windows
        wins = [(s, s+256) for s in range(0, 300-256+1, 32)]; wc = []; wpoles = []
        for s, e in wins:
            Zw = std(Zi[:, s:e]); dw, _ = d_hankel(Zw, L0, rng); wc.append(dw); wpoles.append(classify(esprit(Zw, L0, max(dw, 2)), 256))
        step = np.linalg.norm(np.diff(Zi, axis=1), axis=0); out["windowed_d_hankel"] = wc; out["windowed_real_poles"] = [p["real"] for p in wpoles]; out["windowed_unresolved"] = [p["unresolved"] for p in wpoles]
        out["step_norm_first50_last50"] = [float(step[:50].mean()), float(step[-50:].mean())]; out["dist_to_final_first_last"] = [float(np.linalg.norm(Zi[:, 0]-Zi[:, -1])), float(np.linalg.norm(Zi[:, -51]-Zi[:, -1]))]
        res["rollouts"].append(out)
        print(f"rollout {k}: d_Hankel={out['d_hankel']} resolved={out['pairs_resolved']} unresolved={out['pairs_unresolved']} {out['periods_unresolved']} real={out['real_poles']}; MC-SSA above AR1 {out['above_AR1']} / 1/f {out['above_1/f']}; windowed d {wc} real poles {out['windowed_real_poles']} unresolved {out['windowed_unresolved']}; step {out['step_norm_first50_last50']}; dist to final {out['dist_to_final_first_last']}")
    json.dump(res, open(os.path.join(HERE, "medium_SG_poles.json"), "w"), indent=1)
