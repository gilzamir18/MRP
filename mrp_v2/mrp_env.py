"""
mrp_env.py — Multi-Resource Problem (MRP), revised implementation (v2).

Dynamics are kept identical to the original ``env.py`` (HomeostaticEnv) so that
task difficulty is comparable, with the following corrections requested/implied
by the review:

  * Pairing: every random draw (initial state, initial decay, decay random walk,
    observation noise) comes from per-episode generators seeded by ``reset(seed)``.
    Dynamics and observation noise use *separate* streams, so the exogenous
    process for a given evaluation seed is identical for every method.
  * Horizon: episodes are truncated after exactly ``max_steps`` (=1000) steps
    (the original code allowed 1001).
  * Observation noise: by default added to the state variables only (paper Eq. 28);
    ``noise_on="all"`` reproduces the original behaviour (noise on all 4n+1 entries).
  * Reward source: reward is computed from the latent state (default). For the
    noise-mechanism test, ``reward_source="observed"`` computes it from the noisy
    observation of the state instead.
  * Decay process: as in the original code, per-variable decay rates are sampled
    at reset and then follow a Gaussian random walk (``decay_walk=True``).
    NOTE: the manuscript currently says the rates are "held fixed within the
    episode"; the text must be corrected (or ``decay_walk=False`` used).

The policy outputs unbounded allocation logits; allocation = b * softmax(logits).
"""
from __future__ import annotations

import numpy as np

MIN_SIGNIFICANT = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions.  All take per-variable absolute deviations before (d0) and
# after (d1) the transition, the OR parameters (z, beta) and return a scalar.
# All are averaged over variables (paper Eq. 15) unless the aggregation says
# otherwise.
# ─────────────────────────────────────────────────────────────────────────────

def or_per_variable(d0, d1, z, beta):
    u = (d1 <= z).astype(np.float64)
    m = (d1 >= d0).astype(np.float64)
    return u + (1.0 - u) * (1.0 - (1.0 + beta) * m)


def reward_or(d0, d1, z, beta, agg="mean", tau=0.2):
    r = or_per_variable(d0, d1, z, beta)
    if agg == "mean":
        return float(r.mean())
    if agg == "min":
        return float(r.min())
    if agg == "softmin":
        # soft-min: -tau * log mean exp(-r/tau); between min (tau->0) and mean (tau->inf)
        x = -r / tau
        mx = x.max()
        return float(-tau * (mx + np.log(np.mean(np.exp(x - mx)))))
    raise ValueError(agg)


def reward_dr(d0, d1, **_):
    return float(-np.mean(d1 ** 2))


def reward_dq(d0, d1, **_):
    return float(np.mean(d0 ** 2 - d1 ** 2))


def reward_deadzone(d0, d1, z=0.5, **_):
    return float(-np.mean(np.maximum(0.0, d1 - z) ** 2))


# Variant registry ------------------------------------------------------------
# kind: "or" (uses z, beta, agg) or a continuous function; "alive": constant
# per-step bonus added to the reward.
REWARD_SPECS = {
    # main comparison
    "OR_adapt":   dict(kind="or", agg="mean", learn=True),
    "OR_fixed":   dict(kind="or", agg="mean", z=0.5, beta=1.0),
    "DQ":         dict(kind="dq"),
    "DR":         dict(kind="dr"),
    # reward sign / offset controls
    "Alive":      dict(kind="zero", alive=1.0),
    "DR_alive":   dict(kind="dr", alive=1.0),
    "DQ_alive":   dict(kind="dq", alive=1.0),
    # regime-geometry controls
    "DeadZone":   dict(kind="deadzone", z=0.5),
    "OR_noZone":  dict(kind="or", agg="mean", z=0.0, beta=1.0),
    "OR_noPen":   dict(kind="or", agg="mean", z=0.5, beta=0.0),
    "OR_min":     dict(kind="or", agg="min", z=0.5, beta=1.0),
    "OR_softmin": dict(kind="or", agg="softmin", z=0.5, beta=1.0),
}


def make_reward_fn(variant: str, fixed_z=None, fixed_beta=None):
    """Return reward(d0, d1, z=None, beta=None) for a registered variant.

    ``fixed_z`` / ``fixed_beta`` override the fixed OR parameters (used by the
    calibration study, e.g. OR fixed at the values learned by Adaptive OR).
    """
    spec = dict(REWARD_SPECS[variant])
    if fixed_z is not None:
        spec["z"] = fixed_z
    if fixed_beta is not None:
        spec["beta"] = fixed_beta
    alive = spec.get("alive", 0.0)
    kind = spec["kind"]

    if kind == "or":
        agg = spec["agg"]
        z_fix, b_fix = spec.get("z"), spec.get("beta")

        def fn(d0, d1, z=None, beta=None):
            zz = z_fix if z is None else z
            bb = b_fix if beta is None else beta
            return reward_or(d0, d1, zz, bb, agg=agg) + alive
        return fn
    if kind == "dr":
        return lambda d0, d1, z=None, beta=None: reward_dr(d0, d1) + alive
    if kind == "dq":
        return lambda d0, d1, z=None, beta=None: reward_dq(d0, d1) + alive
    if kind == "deadzone":
        zz = spec["z"]
        return lambda d0, d1, z=None, beta=None: reward_deadzone(d0, d1, z=zz) + alive
    if kind == "zero":
        return lambda d0, d1, z=None, beta=None: 0.0 + alive
    raise ValueError(kind)


class MRPEnv:
    """Minimal (non-gymnasium) MRP environment for fast single-process training."""

    def __init__(self, n: int, budget: float, obs_noise: float, reward_fn,
                 decay_mu: float = 1e-3, decay_std: float = 1e-3,
                 decay_walk: bool = True, max_steps: int = 1000,
                 noise_on: str = "state", reward_source: str = "latent",
                 safety_zone_obs: float = 0.5):
        self.n = n
        self.budget = float(budget)
        self.obs_noise = float(obs_noise)
        self.reward_fn = reward_fn
        self.decay_mu = decay_mu
        self.decay_std = decay_std
        self.decay_walk = decay_walk
        self.max_steps = max_steps
        self.noise_on = noise_on
        self.reward_source = reward_source
        self.safety_zone_obs = safety_zone_obs
        self.obs_dim = 4 * n + 1
        self.act_dim = n
        self._rng_dyn = None
        self._rng_obs = None

    # ------------------------------------------------------------------
    def reset(self, seed: int):
        ss = np.random.SeedSequence(int(seed))
        s_dyn, s_obs = ss.spawn(2)
        self._rng_dyn = np.random.default_rng(s_dyn)
        self._rng_obs = np.random.default_rng(s_obs)
        self.v = self._rng_dyn.uniform(-0.9, 0.9, self.n)
        self.decay = self._rng_dyn.normal(self.decay_mu, self.decay_std, self.n)
        self.t = 0
        self._obs_v = self._noisy_state()
        return self._obs()

    def _noisy_state(self):
        if self.obs_noise > 0.0:
            return self.v + self._rng_obs.normal(0.0, self.obs_noise, self.n)
        return self.v.copy()

    def _obs(self):
        n = self.n
        obs = np.concatenate([
            self._obs_v,
            np.ones(n),                         # survival bounds
            np.full(n, self.safety_zone_obs),   # safety zones (constant)
            self.decay,                         # current decay rates
            [self.budget],
        ])
        if self.noise_on == "all" and self.obs_noise > 0.0:
            noise = self._rng_obs.normal(0.0, self.obs_noise, obs.shape)
            noise[:n] = 0.0          # state part already noisy
            obs = obs + noise
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def allocation(self, logits):
        x = np.asarray(logits, dtype=np.float64)
        x = x - x.max()
        e = np.exp(x)
        s = e.sum()
        if not np.isfinite(s) or s <= 0:
            return np.full(self.n, self.budget / self.n)
        return self.budget * e / s

    def step(self, logits, z=None, beta=None):
        a = self.allocation(logits)
        v_prev = self.v
        obs_prev = self._obs_v
        if self.decay_walk and self.decay_std > 0:
            self.decay = self.decay + self._rng_dyn.normal(0.0, self.decay_std, self.n)
        v = v_prev.copy()
        neg = v < -MIN_SIGNIFICANT
        pos = v > MIN_SIGNIFICANT
        v = np.where(neg, v - self.decay, np.where(pos, v + self.decay, v))
        neg = v < -MIN_SIGNIFICANT
        pos = v > MIN_SIGNIFICANT
        v = np.where(neg, v + a, np.where(pos, v - a, v))
        self.v = v
        self.t += 1
        self._obs_v = self._noisy_state()

        if self.reward_source == "observed":
            d0, d1 = np.abs(obs_prev), np.abs(self._obs_v)
        else:
            d0, d1 = np.abs(v_prev), np.abs(v)
        r = self.reward_fn(d0, d1, z=z, beta=beta)

        dev = np.abs(v)
        terminated = bool(np.any(dev > 1.0))
        truncated = (not terminated) and self.t >= self.max_steps
        info = {"alloc": a, "dev_prev": np.abs(v_prev), "dev": dev}
        return self._obs(), r, terminated, truncated, info
