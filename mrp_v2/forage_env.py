"""
forage_env.py — vectorised homeostatic foraging gridworld (replacement for Crafter).

E parallel episodes on a W x W grid.  The agent has k internal variables v_i
(set point 0, viability |v_i| <= 1).  Each variable depletes at its own rate
(v_i -= lambda_i per step; lambda_i drawn per episode and following a small
random walk, as in the MRP).  Resource i is available at one station whose
position is drawn per episode.  The "consume" action at station i adds `gain`
to v_i, so a variable can be over-consumed (v_i > 0): deviation is two-sided.

Actions (discrete): 0 up, 1 down, 2 left, 3 right, 4 stay, 5 consume (= stay elsewhere).
Observation: [v_1..v_k (optionally noisy), (dx_i, dy_i)/(W-1) for each station,
on-station indicator for each resource]  -> 4k entries.

Termination: any |v_i| > 1 ends the episode (terminate=True).  With
terminate=False (no-termination control, as stage NT of the MRP) v is clipped to
[-1, 1] and every episode lasts max_steps, so a constant reward offset carries
no information.

Pairing: every episode has its own generators (dynamics / observation noise),
seeded by reset(i, seed), so evaluation episodes are identical across methods.
"""
from __future__ import annotations

import numpy as np

MOVES = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0], [0, 0]], dtype=np.int64)
STAY, CONSUME = 4, 5
N_ACTIONS = 6


# ── vectorised reward functions: d0, d1 have shape (E, k); return (E,) ──────────
def or_per_variable(d0, d1, z, beta):
    z = np.asarray(z, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if beta.ndim == 1:
        beta = beta[:, None]
    u = (d1 <= z).astype(np.float64)
    m = (d1 >= d0).astype(np.float64)
    return u + (1.0 - u) * (1.0 - (1.0 + beta) * m)


REWARD_SPECS = {
    "OR_adapt":  dict(kind="or", learn=True),
    "OR_fixed":  dict(kind="or", z=0.5, beta=1.0),
    "DQ":        dict(kind="dq"),
    "DR":        dict(kind="dr"),
    "Alive":     dict(kind="zero", alive=1.0),
    "DR_alive":  dict(kind="dr", alive=1.0),
    "DeadZone":  dict(kind="deadzone", z=0.5),
    "OR_noZone": dict(kind="or", z=0.0, beta=1.0),
    "OR_noPen":  dict(kind="or", z=0.5, beta=0.0),
    # sign controls on the scale of DQ (2026-11-16; see docs/prereg_2026-11-16.md)
    "DQ_c001":   dict(kind="dq", alive=0.01),
    "DQ_c01":    dict(kind="dq", alive=0.1),
    "OR_m1":     dict(kind="or", z=0.5, beta=1.0, alive=-1.0),
}


def make_reward_fn(variant):
    if variant.startswith("ORg_"):               # fixed OR on a (z, beta) grid: ORg_z0.30_b2.00
        _, zs, bs = variant.split("_")
        spec = dict(kind="or", z=float(zs[1:]), beta=float(bs[1:]))
    elif variant == "OR_outer":                  # z, beta supplied per episode by the outer loop
        spec = dict(kind="or")
    else:
        spec = REWARD_SPECS[variant]
    alive, kind = spec.get("alive", 0.0), spec["kind"]
    if kind == "or":
        zf, bf = spec.get("z"), spec.get("beta")

        def fn(d0, d1, z=None, beta=None):
            zz = zf if z is None else z
            bb = bf if beta is None else beta
            return or_per_variable(d0, d1, zz, bb).mean(1) + alive
        return fn
    if kind == "dr":
        return lambda d0, d1, z=None, beta=None: -(d1 ** 2).mean(1) + alive
    if kind == "dq":
        return lambda d0, d1, z=None, beta=None: (d0 ** 2 - d1 ** 2).mean(1) + alive
    if kind == "deadzone":
        zz = spec["z"]
        return lambda d0, d1, z=None, beta=None: -(np.maximum(0.0, d1 - zz) ** 2).mean(1) + alive
    if kind == "zero":
        return lambda d0, d1, z=None, beta=None: np.zeros(d0.shape[0]) + alive
    raise ValueError(kind)


class ForageEnv:
    def __init__(self, n_envs, k, reward_fn, W=9, decay_mu=0.008, decay_cv=0.25,
                 decay_walk=0.0002, gain=0.2, obs_noise=0.0, max_steps=500,
                 terminate=True, v0_low=-0.5, v0_high=0.2):
        self.E, self.k, self.W = int(n_envs), int(k), int(W)
        self.reward_fn = reward_fn
        self.decay_mu, self.decay_cv, self.decay_walk = decay_mu, decay_cv, decay_walk
        self.gain, self.obs_noise = gain, obs_noise
        self.max_steps, self.terminate = max_steps, terminate
        self.v0_low, self.v0_high = v0_low, v0_high
        self.obs_dim, self.n_actions = 4 * self.k, N_ACTIONS
        E, k = self.E, self.k
        self.v = np.zeros((E, k))
        self.decay = np.zeros((E, k))
        self.pos = np.zeros((E, 2), np.int64)
        self.st = np.zeros((E, k, 2), np.int64)
        self.t = np.zeros(E, np.int64)
        self.obs_v = np.zeros((E, k))
        self.rd = [None] * E
        self.ro = [None] * E

    # ------------------------------------------------------------------
    def reset_one(self, i, seed):
        s_dyn, s_obs = np.random.SeedSequence(int(seed)).spawn(2)
        rd = self.rd[i] = np.random.default_rng(s_dyn)
        self.ro[i] = np.random.default_rng(s_obs)
        W, k = self.W, self.k
        cells = rd.choice(W * W, size=k + 1, replace=False)      # distinct cells
        xy = np.stack([cells % W, cells // W], 1)
        self.pos[i], self.st[i] = xy[0], xy[1:]
        self.v[i] = rd.uniform(self.v0_low, self.v0_high, k)
        self.decay[i] = np.maximum(1e-4, rd.normal(self.decay_mu, self.decay_cv * self.decay_mu, k))
        self.t[i] = 0
        self.obs_v[i] = self._noisy(i)

    def _noisy(self, i):
        if self.obs_noise > 0:
            return self.v[i] + self.ro[i].normal(0.0, self.obs_noise, self.k)
        return self.v[i].copy()

    def obs(self):
        rel = (self.st - self.pos[:, None, :]) / (self.W - 1)              # (E,k,2)
        on = (rel == 0).all(2).astype(np.float64)                          # (E,k)
        return np.concatenate([self.obs_v, rel.reshape(self.E, -1), on], 1).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, actions, z=None, beta=None):
        a = np.asarray(actions, np.int64)
        E, k = self.E, self.k
        if self.decay_walk > 0:
            for i in range(E):
                self.decay[i] += self.rd[i].normal(0.0, self.decay_walk, k)
            np.maximum(self.decay, 1e-4, out=self.decay)
        v_prev = self.v.copy()
        # movement
        self.pos = np.clip(self.pos + MOVES[a], 0, self.W - 1)
        # consumption
        on = (self.st == self.pos[:, None, :]).all(2)                     # (E,k)
        eat = on & (a == CONSUME)[:, None]
        v = v_prev - self.decay + self.gain * eat
        viol = np.abs(v) > 1.0
        if not self.terminate:
            v = np.clip(v, -1.0, 1.0)
        self.v = v
        self.t += 1
        for i in range(E):
            self.obs_v[i] = self._noisy(i)
        r = self.reward_fn(np.abs(v_prev), np.abs(v), z=z, beta=beta)
        term = viol.any(1) if self.terminate else np.zeros(E, bool)
        trunc = (~term) & (self.t >= self.max_steps)
        info = {"viol": viol, "dev": np.abs(v), "dev_prev": np.abs(v_prev), "eat": eat,
                "over": eat & (v_prev >= 0.0)}
        return r, term, trunc, info


# ── reference policies (operate on the env state directly) ───────────────────
def policy_random(env, rng):
    return rng.integers(0, N_ACTIONS, env.E)


class GreedyPolicy:
    """Go to the station of the most depleted variable and consume until it reaches
    `stop`; then switch to the (new) most depleted variable.  Uses the observed v."""

    def __init__(self, stop=0.2):
        self.stop, self.tgt = stop, None

    def __call__(self, env, rng=None):
        E, idx = env.E, np.arange(env.E)
        if self.tgt is None or len(self.tgt) != E:
            self.tgt = np.argmin(env.obs_v, 1)
        done = env.obs_v[idx, self.tgt] >= self.stop
        self.tgt = np.where(done, np.argmin(env.obs_v, 1), self.tgt)
        d = env.st[idx, self.tgt] - env.pos
        act = np.where(d[:, 0] > 0, 3, np.where(d[:, 0] < 0, 2, np.where(d[:, 1] > 0, 1, 0)))
        need = env.obs_v[idx, self.tgt] < self.stop
        return np.where((d == 0).all(1), np.where(need, CONSUME, STAY), act)

    def reset(self):
        self.tgt = None


def run_reference(policy, k, cond_kw, seeds, terminate=True, rng_seed=0):
    """Evaluate a reference policy on paired episodes (one env per seed)."""
    env = ForageEnv(len(seeds), k, make_reward_fn("Alive"), terminate=terminate, **cond_kw)
    for i, s in enumerate(seeds):
        env.reset_one(i, s)
    if hasattr(policy, "reset"):
        policy.reset()
    rng = np.random.default_rng(rng_seed)
    return rollout_eval(env, lambda e: policy(e, rng))


def rollout_eval(env, act_fn):
    """Run each of env.E episodes once to the end; return per-episode metrics."""
    E, k = env.E, env.k
    alive = np.ones(E, bool)
    L = np.zeros(E, np.int64)
    first_viol = np.full(E, -1)
    died = np.zeros(E, bool)
    acc = {m: np.zeros(E) for m in ("viable", "all_viable", "in_zone", "dev", "eat", "over")}
    while alive.any():
        a = act_fn(env)
        _, term, trunc, info = env.step(a)
        al = alive.copy()
        L += al
        vi = info["viol"]
        acc["viable"] += al * (1 - vi.mean(1))
        acc["all_viable"] += al * (~vi.any(1))
        acc["in_zone"] += al * (info["dev"] <= 0.5).mean(1)
        acc["dev"] += al * info["dev"].mean(1)
        acc["eat"] += al * info["eat"].any(1)
        acc["over"] += al * info["over"].any(1)
        newv = al & vi.any(1) & (first_viol < 0)
        first_viol[newv] = L[newv]
        died |= al & term
        alive &= ~(term | trunc)
    first_viol = np.where(first_viol < 0, L, first_viol)
    out = []
    for i in range(E):
        n = max(L[i], 1)
        out.append({"length": int(L[i]), "event": bool(died[i]),
                    "first_viol": int(first_viol[i]),
                    "frac_viable": acc["viable"][i] / n, "frac_all_viable": acc["all_viable"][i] / n,
                    "frac_in_zone": acc["in_zone"][i] / n, "mean_dev": acc["dev"][i] / n,
                    "eat_rate": acc["eat"][i] / n,
                    "over_share": acc["over"][i] / max(acc["eat"][i], 1)})
    return out
