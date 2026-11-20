"""BOID for JEPA, toy reproduction.  A top-down U-maze with a point agent and a pushable block, one uniform-random
exploration trajectory, 48x48 frames.  Three world models with the same encoder and predictor, differing only in the
objective, after Ren, Wang, Hoang, Kuang, LeCun (2026, AGL-TR-2026-04):
  SG        prediction loss with a stop-gradient target
  SVReg     prediction loss without stop-gradient + lambda * E_u max(0, 1 - std_t(z.u))^2, u uniform on the sphere (Eq. 1)
  SlowSVReg SVReg + beta * E_t |b_{t+1}-b_t|^2 / (2 tr Var b) on the last 1/8 of the latent (Eq. 2),
            with the gated residual predictor z_{t+1} = z_t + c * f(z_t, a_t), gate c initialised at 0.1 (Eq. 3)
Measured on held-out trajectories of the latent z_t: the BOID count (MSSA with permutation surrogates), the poles of the
leading modes, the share of latent power at periods under five steps (their jitter), the linear readout R^2 of the agent
and block positions (content), the latent path jaggedness (geometry), and the count on autonomous rollouts of the
predictor beyond its training horizon.  Usage: python jepa_boid_toy.py [steps] [seeds]"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, time, json, os
torch.set_num_threads(4); HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000; SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
IMG, D, T, B = 48, 32, 16, 32; LAMBDA, BETA = 10.0, 0.1; SLOW = D//8

# ----------------------------------------------------------------- environment: U-maze with a pushable block
WALLS = [(0.0, 0.0, 1.0, 0.06), (0.0, 0.94, 1.0, 0.06), (0.0, 0.0, 0.06, 1.0), (0.94, 0.0, 0.06, 1.0), (0.30, 0.35, 0.40, 0.30)]  # x,y,w,h; central block = the U's core
def blocked(p, r=0.04):
    for (x, y, w, h) in WALLS:
        if x - r < p[0] < x + w + r and y - r < p[1] < y + h + r: return True
    return False
def free_point(rng):
    while True:
        p = rng.uniform(0.08, 0.92, 2)
        if not blocked(p, 0.06): return p
def rollout(N, rng, step=0.05, actions=None):
    """uniform-random actions (velocity commands), five env steps per frame; the block moves only when pushed"""
    agent = free_point(rng); block = free_point(rng)
    while np.linalg.norm(block - agent) < 0.2: block = free_point(rng)
    A, P, Q = [], [], []
    for n in range(N):
        a = actions[n] if actions is not None else rng.uniform(-1, 1, 2)
        for _ in range(5):
            new = agent + step/5*a
            if not blocked(new): agent = new
            d = block - agent
            if np.linalg.norm(d) < 0.09:                                     # contact: push the block away from the agent
                nb = block + 0.6*(step/5)*(d/ (np.linalg.norm(d)+1e-9))*np.linalg.norm(a)
                if not blocked(nb, 0.05): block = nb
        A.append(a.copy()); P.append(agent.copy()); Q.append(block.copy())
    return np.array(A, np.float32), np.array(P, np.float32), np.array(Q, np.float32)
def render(P, Q):
    yy, xx = np.mgrid[0:IMG, 0:IMG]/(IMG-1); frames = np.zeros((len(P), IMG, IMG), np.float32)
    wall = np.zeros((IMG, IMG), np.float32)
    for (x, y, w, h) in WALLS: wall[(xx >= x) & (xx <= x+w) & (yy >= y) & (yy <= y+h)] = 0.35
    for i in range(len(P)):
        f = wall.copy()
        f += np.exp(-(((xx-P[i, 0])**2 + (yy-P[i, 1])**2))/(2*0.03**2))             # agent: bright blob
        f += 0.6*np.exp(-(((xx-Q[i, 0])**2 + (yy-Q[i, 1])**2))/(2*0.035**2))        # block: dimmer blob
        frames[i] = np.clip(f, 0, 1)
    return frames

# ----------------------------------------------------------------- models
class Encoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.GELU(), nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
                              nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.GELU(), nn.Flatten(), nn.Linear(64*6*6, 128), nn.GELU(), nn.Linear(128, D))
    def forward(s, x): return s.net(x)
class Predictor(nn.Module):
    """3-frame history + action; gated residual form when gated=True"""
    def __init__(s, gated):
        super().__init__(); s.gated = gated
        s.f = nn.Sequential(nn.Linear(3*D + 2, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, D))
        if gated: s.c = nn.Parameter(torch.full((D,), 0.1))
    def forward(s, z3, a):
        out = s.f(torch.cat([z3.flatten(1), a], 1))
        return z3[:, -1] + s.c*out if s.gated else out

def svreg(z, n_dir=64):
    u = F.normalize(torch.randn(n_dir, z.shape[-1]), dim=1); proj = z.reshape(-1, z.shape[-1]) @ u.T
    return (F.relu(1 - proj.std(0))**2).mean()
def slow_term(b):                                                               # b: (B, T, SLOW)
    num = ((b[:, 1:] - b[:, :-1])**2).sum(-1).mean(); var = b.reshape(-1, SLOW).var(0).sum()
    return num/(2*var + 1e-8)

def train(variant, frames, actions, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    enc, pred = Encoder(), Predictor(gated=(variant == "SlowSVReg"))
    opt = torch.optim.Adam([{"params": enc.parameters(), "lr": 1e-3}, {"params": pred.parameters(), "lr": 5e-4}])
    X = torch.tensor(frames)[:, None]; A = torch.tensor(actions); N = len(frames); t0 = time.time()
    for it in range(STEPS):
        idx = rng.integers(0, N - T - 1, B); xb = torch.stack([X[i:i+T+1] for i in idx]); ab = torch.stack([A[i:i+T] for i in idx])
        z = enc(xb.reshape(-1, 1, IMG, IMG)).reshape(B, T+1, D)
        hist = torch.stack([z[:, max(t-2, 0)] for t in range(T)], 1), torch.stack([z[:, max(t-1, 0)] for t in range(T)], 1), z[:, :T]
        z3 = torch.stack(hist, 2)                                              # (B, T, 3, D)
        zhat = pred(z3.reshape(B*T, 3, D), ab.reshape(B*T, 2)).reshape(B, T, D)
        target = z[:, 1:].detach() if variant == "SG" else z[:, 1:]
        loss = F.mse_loss(zhat, target)
        if variant in ("SVReg", "SlowSVReg"): loss = loss + LAMBDA*svreg(z)
        if variant == "SlowSVReg": loss = loss + BETA*slow_term(z[:, :, -SLOW:])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0 or it == STEPS-1: print(f"    {variant} seed {seed} step {it} loss {loss.item():.4f} ({time.time()-t0:.0f} s)", flush=True)
    enc.eval(); pred.eval(); return enc, pred

# ----------------------------------------------------------------- measurements on the latent
def block_hankel(Z, L):                                                         # Z: channels x N
    return np.vstack([np.lib.stride_tricks.sliding_window_view(Z[j], L).T for j in range(Z.shape[0])])
def gram_eigs(Z, L):
    H = block_hankel(Z, L); K = H.shape[1]; G = H @ H.T if H.shape[0] <= K else H.T @ H
    return np.sort(np.linalg.eigvalsh(G))[::-1]/K
def boid_count(Z, L=16, Bsur=40, eta=0.05, rng=None):
    """components above the 95% permutation surrogate, as in the real-data sections of the MSSA papers"""
    rng = rng or np.random.default_rng(0); ell = gram_eigs(Z, L)
    sur = np.array([gram_eigs(np.array([rng.permutation(x) for x in Z]), L) for _ in range(Bsur)]); q = np.quantile(sur, 1-eta, axis=0)
    n = min(len(ell), len(q)); return int(np.sum(ell[:n] > q[:n])), ell
def esprit_freqs(Z, L, d):
    H = block_hankel(Z, L); U = np.linalg.svd(H, full_matrices=False)[0][:, :d]; m = Z.shape[0]; Bk = U.reshape(m, L, d)
    up, dn = Bk[:, :-1].reshape(m*(L-1), d), Bk[:, 1:].reshape(m*(L-1), d); z = np.linalg.eigvals(np.linalg.lstsq(up, dn, rcond=None)[0])
    f = np.abs(np.angle(z))/(2*np.pi); keep = np.angle(z) > 0; return np.sort(f[keep]), np.abs(z[keep])
def jitter(Z):
    """share of latent power at periods under five steps (their early-phase jitter measure), variance-weighted over channels"""
    Zc = Z - Z.mean(1, keepdims=True); P = np.abs(np.fft.rfft(Zc, axis=1))**2; f = np.fft.rfftfreq(Z.shape[1]); hi = f > 0.2
    return float(P[:, hi].sum()/P[:, 1:].sum())
def jaggedness(Z):
    """path length of the latent route divided by the length of its 10-step-smoothed version (1 = smooth)"""
    d = np.linalg.norm(np.diff(Z, axis=1), axis=0).sum(); k = 10; S = np.stack([np.convolve(z, np.ones(k)/k, mode="valid") for z in Z]); ds = np.linalg.norm(np.diff(S, axis=1), axis=0).sum()
    return float(d/max(ds, 1e-9))
def readout_r2(Z, Y):
    """linear readout of positions from the latent, R^2 on the second half after fitting on the first"""
    n = Z.shape[1]//2; Xa = np.c_[Z.T, np.ones(Z.shape[1])]; W = np.linalg.lstsq(Xa[:n], Y[:n], rcond=None)[0]; pred = Xa[n:] @ W
    return float(1 - ((Y[n:] - pred)**2).sum()/((Y[n:] - Y[n:].mean(0))**2).sum())
@torch.no_grad()
def latents(enc, frames): return enc(torch.tensor(frames)[:, None]).numpy().T             # D x N
@torch.no_grad()
def imagine(enc, pred, frames, actions, steps=400):
    """autonomous rollout of the predictor from a real 3-frame history, actions held at zero: the dynamics beyond the horizon"""
    z = enc(torch.tensor(frames[:3])[:, None]); hist = [z[0], z[1], z[2]]; out = []
    for t in range(steps):
        z3 = torch.stack(hist[-3:], 0)[None]; zn = pred(z3, torch.zeros(1, 2))[0]; hist.append(zn); out.append(zn.numpy())
    return np.array(out).T

if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(0)
    A_tr, P_tr, Q_tr = rollout(5000, rng); X_tr = render(P_tr, Q_tr)
    held = [rollout(1024, np.random.default_rng(100+k)) for k in range(3)]; Xh = [render(P, Q) for (_, P, Q) in held]
    print(f"data: 5000 training frames, 3 held-out runs of 1024 frames; block moved in {np.mean([np.linalg.norm(np.diff(Q, axis=0), axis=1).sum() > 0.05 for (_, _, Q) in held]):.0%} of runs ({time.time()-t0:.0f} s)")
    results = {}
    for variant in ("SG", "SVReg", "SlowSVReg"):
        rows = []
        for seed in range(SEEDS):
            enc, pred = train(variant, X_tr, A_tr, seed)
            r = dict(count=[], count_slow=[], jitter=[], jag=[], r2_agent=[], r2_block=[], f_lead=[], count_imag=[], var_slow_share=[], lat_std=[])
            for (A, P, Q), X in zip(held, Xh):
                Z = latents(enc, X); Zs = (Z - Z.mean(1, keepdims=True))/(Z.std(1, keepdims=True) + 1e-8)
                c, ell = boid_count(Zs); r["count"].append(c)
                f, mod = esprit_freqs(Zs, 16, max(2, min(c, 16))); r["f_lead"].append(f[:4].tolist())
                r["jitter"].append(jitter(Z)); r["jag"].append(jaggedness(Z)); r["lat_std"].append(float(Z.std()))
                r["r2_agent"].append(readout_r2(Z, P)); r["r2_block"].append(readout_r2(Z, Q))
                r["var_slow_share"].append(float(Z[-SLOW:].var(1).sum()/Z.var(1).sum()))
                if variant == "SlowSVReg": r["count_slow"].append(boid_count(Zs[-SLOW:])[0])
                Zi = imagine(enc, pred, X, A); Zi = (Zi - Zi.mean(1, keepdims=True))/(Zi.std(1, keepdims=True) + 1e-8); r["count_imag"].append(boid_count(Zi)[0])
            # save latents, positions, imagined rollouts and the models for the figures
            np.savez_compressed(os.path.join(HERE, f"latents_{variant}_seed{seed}.npz"),
                                **{f"Z{k}": latents(enc, X) for k, X in enumerate(Xh)}, **{f"P{k}": P for k, (A, P, Q) in enumerate(held)},
                                **{f"Q{k}": Q for k, (A, P, Q) in enumerate(held)}, **{f"Zimag{k}": imagine(enc, pred, X, A) for k, ((A, P, Q), X) in enumerate(zip(held, Xh))})
            torch.save({"enc": enc.state_dict(), "pred": pred.state_dict()}, os.path.join(HERE, f"model_{variant}_seed{seed}.pt"))
            rows.append(r); print(f"  {variant} seed {seed}: count {r['count']} | imagined {r['count_imag']} | jitter {np.mean(r['jitter']):.3f} | jag {np.mean(r['jag']):.2f} | R2 agent {np.mean(r['r2_agent']):.2f} block {np.mean(r['r2_block']):.2f} | lead f {np.round(np.mean(r['f_lead'], 0), 3).tolist()}", flush=True)
        results[variant] = rows
    def agg(v, key): return (float(np.mean([np.mean(r[key]) for r in results[v]])), float(np.std([np.mean(r[key]) for r in results[v]])))
    print("\n=== BOID for JEPA (toy U-maze with a pushable block; mean ± sd over seeds, each averaged over 3 held-out runs) ===")
    print(f"{'variant':10s} {'BOID count':>12s} {'count, rollout':>15s} {'jitter':>8s} {'jaggedness':>11s} {'R2 agent':>9s} {'R2 block':>9s} {'slow-ch var share':>18s}")
    for v in results:
        c, cs = agg(v, "count"); ci, cis = agg(v, "count_imag"); j, js = agg(v, "jitter"); g, gs = agg(v, "jag"); ra, ras = agg(v, "r2_agent"); rb, rbs = agg(v, "r2_block"); vs, vss = agg(v, "var_slow_share")
        print(f"{v:10s} {c:6.1f}±{cs:4.1f}   {ci:6.1f}±{cis:4.1f}   {j:5.3f}±{js:.3f} {g:6.2f}±{gs:.2f} {ra:5.2f}±{ras:.2f} {rb:5.2f}±{rbs:.2f}   {vs:5.3f}±{vss:.3f}")
    if "SlowSVReg" in results: print("SlowSVReg: count in the slow channels alone:", [r["count_slow"] for r in results["SlowSVReg"]])
    json.dump(results, open(os.path.join(HERE, "jepa_boid_results.json"), "w"), indent=1)
    print(f"elapsed {time.time()-t0:.0f} s")
