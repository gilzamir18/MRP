"""
forage_ppo.py — PPO-Clip with a categorical policy over vectorised ForageEnv.

Same algorithm for every reward variant.  Adaptive OR adds the four
state-independent scalars (mu_sz, log sigma_sz, mu_beta, log sigma_beta) of the
MRP implementation (z = 0.8 sigmoid(z~) + 0.1, beta = softplus(beta~)); z~ and
beta~ are sampled per environment step and enter the PPO likelihood.

Defaults (CleanRL-style discrete PPO): separate 2x64 tanh MLPs, orthogonal init,
16 envs x 128 steps per update, 4 epochs, 4 minibatches, Adam lr 3e-4 (eps 1e-5),
gamma 0.99, GAE 0.95, clip 0.2, vf 0.5, entropy 0.01, grad-norm 0.5, per-minibatch
advantage normalisation; time-limit truncation is bootstrapped.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from forage_env import ForageEnv, rollout_eval

Normal.set_default_validate_args(False)
Categorical.set_default_validate_args(False)
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _mlp(inp, hid, out, gain):
    net = nn.Sequential(nn.Linear(inp, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, out))
    lin = [l for l in net if isinstance(l, nn.Linear)]
    for i, l in enumerate(lin):
        nn.init.orthogonal_(l.weight, gain=gain if i == len(lin) - 1 else math.sqrt(2))
        nn.init.zeros_(l.bias)
    return net


class Agent(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64, learn_or=False, z_init=0.5, beta_init=1.0,
                 log_sigma_init=-1.0):
        super().__init__()
        self.pi = _mlp(obs_dim, hidden, n_actions, 0.01)
        self.vf = _mlp(obs_dim, hidden, 1, 1.0)
        self.learn_or = learn_or
        if learn_or:
            p = (z_init - 0.1) / 0.8
            self.mu_sz = nn.Parameter(torch.tensor([math.log(p / (1 - p))]))
            self.log_sig_sz = nn.Parameter(torch.tensor([log_sigma_init]))
            self.mu_beta = nn.Parameter(torch.tensor([math.log(math.expm1(beta_init))]))
            self.log_sig_beta = nn.Parameter(torch.tensor([log_sigma_init]))

    # numpy inference --------------------------------------------------
    def sync(self):
        with torch.no_grad():
            g = lambda net: [(l.weight.numpy().T.astype(np.float64), l.bias.numpy().astype(np.float64))
                             for l in net if isinstance(l, nn.Linear)]
            self._pi, self._vf = g(self.pi), g(self.vf)
            if self.learn_or:
                self._or = (float(self.mu_sz), float(self.log_sig_sz), float(self.mu_beta), float(self.log_sig_beta))

    @staticmethod
    def _f(layers, x):
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < len(layers) - 1:
                x = np.tanh(x)
        return x

    def np_logits(self, obs):
        return self._f(self._pi, obs.astype(np.float64))

    def np_value(self, obs):
        return self._f(self._vf, obs.astype(np.float64))[:, 0]

    def np_act(self, obs, rng, deterministic=False):
        lg = self.np_logits(obs)
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(1, keepdims=True)
        if deterministic:
            a = p.argmax(1)
        else:
            u = rng.random(len(p))[:, None]
            a = np.minimum((p.cumsum(1) < u).sum(1), p.shape[1] - 1)
        lp = np.log(p[np.arange(len(p)), a] + 1e-12)
        return a, lp

    def sample_or(self, E, rng, deterministic=False):
        mz, lsz, mb, lsb = self._or
        if deterministic:
            zr, br = np.full(E, mz), np.full(E, mb)
        else:
            zr = mz + math.exp(lsz) * rng.standard_normal(E)
            br = mb + math.exp(lsb) * rng.standard_normal(E)
        lp = (-0.5 * ((zr - mz) / math.exp(lsz)) ** 2 - lsz - _HALF_LOG_2PI
              - 0.5 * ((br - mb) / math.exp(lsb)) ** 2 - lsb - _HALF_LOG_2PI)
        z = 1.0 / (1.0 + np.exp(-zr)) * 0.8 + 0.1
        beta = np.logaddexp(0.0, br)
        return zr, br, z, beta, lp

    # torch evaluation -------------------------------------------------
    def evaluate(self, obs, act, raw):
        d = Categorical(logits=self.pi(obs))
        lp, ent = d.log_prob(act), d.entropy()
        if self.learn_or:
            dz = Normal(self.mu_sz, self.log_sig_sz.exp())
            db = Normal(self.mu_beta, self.log_sig_beta.exp())
            lp = lp + dz.log_prob(raw[:, 0]) + db.log_prob(raw[:, 1])
        return lp, ent, self.vf(obs).squeeze(-1)

    def or_summary(self):
        if not self.learn_or:
            return None
        with torch.no_grad():
            mz, mb = float(self.mu_sz), float(self.mu_beta)
            return {"mu_sz_raw": mz, "sigma_sz": float(self.log_sig_sz.exp()),
                    "mu_beta_raw": mb, "sigma_beta": float(self.log_sig_beta.exp()),
                    "z_of_mean": 1 / (1 + math.exp(-mz)) * 0.8 + 0.1,
                    "beta_of_mean": math.log1p(math.exp(mb)) if mb < 30 else mb}


class OuterParams:
    """Non-circular adaptation of (z_safe, beta): an outer loop on an EXTERNAL fitness.

    (z~, beta~) are sampled once per episode from N(mu, sigma^2) in the unconstrained space
    (same transforms as Adaptive OR) and held fixed during the episode.  They are NOT part of
    the PPO likelihood, so the policy gradient cannot change them.  After each rollout, mu is
    updated by a score-function (evolution-strategy) step on the fitness of the finished
    episodes -- the designer's criterion (survival / viable fraction), not the OR reward:
        mu <- mu + lr * mean( F_hat * (theta~ - mu) / sigma ),  F_hat = standardised fitness.
    This is an online instance of the optimal-reward problem (Singh, Lewis and Barto, 2010).
    """

    def __init__(self, E, rng, z_init=0.5, beta_init=1.0, sigma=0.3, lr=0.05, min_batch=16):
        p = (z_init - 0.1) / 0.8
        self.mu = np.array([math.log(p / (1 - p)), math.log(math.expm1(beta_init))])
        self.sigma, self.lr, self.min_batch, self.rng = sigma, lr, min_batch, rng
        self.raw = np.zeros((E, 2))
        self.buf_raw, self.buf_fit = [], []
        for i in range(E):
            self.sample(i)

    @staticmethod
    def transform(raw):
        return 1.0 / (1.0 + np.exp(-raw[..., 0])) * 0.8 + 0.1, np.logaddexp(0.0, raw[..., 1])

    def sample(self, i):
        self.raw[i] = self.mu + self.sigma * self.rng.standard_normal(2)

    def current(self):
        return self.transform(self.raw)

    def episode_end(self, i, fitness):
        self.buf_raw.append(self.raw[i].copy())
        self.buf_fit.append(float(fitness))
        self.sample(i)

    def update(self):
        if len(self.buf_fit) < self.min_batch:
            return
        F = np.array(self.buf_fit)
        R = np.array(self.buf_raw)
        if F.std() > 1e-12:
            Fh = (F - F.mean()) / F.std()
            self.mu = self.mu + self.lr * (Fh[:, None] * (R - self.mu) / self.sigma).mean(0)
        self.buf_raw, self.buf_fit = [], []

    def summary(self):
        z, b = self.transform(self.mu)
        return {"mu_sz_raw": float(self.mu[0]), "mu_beta_raw": float(self.mu[1]),
                "z_of_mean": float(z), "beta_of_mean": float(b), "sigma": self.sigma}


def train(env: ForageEnv, ag: Agent, total_timesteps, seed, n_steps=128, n_epochs=4, n_mb=4,
          lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          outer=False, outer_kw=None, freeze_policy=False, norm_reward=False):
    # norm_reward=True: reward scaling by the running SD of the discounted return, per env
    # (as norm_reward in the MRP agent and SB3 VecNormalize): r / sqrt(Var[R] + 1e-8), clipped
    # to +-10.  Added 2026-11-20 (second review); default False leaves earlier runs unchanged.
    torch.manual_seed(seed)
    # freeze_policy=True: the actor network is never updated (behaviour fixed at initialisation);
    # only the critic and, for Adaptive OR, the reward parameters (z, beta) learn.  Used to show
    # that the drift of (z, beta) does not depend on any change in behaviour.
    params = [q for n, q in ag.named_parameters() if not (freeze_policy and n.startswith("pi."))]
    opt = torch.optim.Adam(params, lr=lr, eps=1e-5)
    ag.sync()
    E = env.E
    rng_ep = np.random.default_rng(10_000 + seed)
    rng_act = np.random.default_rng(20_000 + seed)
    for i in range(E):
        env.reset_one(i, int(rng_ep.integers(1 << 31)))
    obs = env.obs()
    ep_len, ep_ret = np.zeros(E, np.int64), np.zeros(E)
    ep_viable = np.zeros(E)
    outer_p = OuterParams(E, np.random.default_rng(30_000 + seed), **(outer_kw or {})) if outer else None
    log = {"ep_end_step": [], "ep_len": [], "ep_ret": [], "or_traj": [], "upd_entropy": []}
    if norm_reward:
        from ppo_agent import RunningMeanStd
        ret_rms, ret_acc = RunningMeanStd(), np.zeros(E)
    steps = 0
    n_upd = max(1, total_timesteps // (n_steps * E))
    for upd in range(n_upd):
        B_o = np.zeros((n_steps, E, env.obs_dim), np.float32)
        B_a = np.zeros((n_steps, E), np.int64)
        B_raw = np.zeros((n_steps, E, 2), np.float32)
        B_lp, B_v, B_r, B_d = (np.zeros((n_steps, E), np.float32) for _ in range(4))
        for t in range(n_steps):
            a, lp = ag.np_act(obs, rng_act)
            v = ag.np_value(obs)
            if ag.learn_or:
                zr, br, z, beta, lpo = ag.sample_or(E, rng_act)
                lp = lp + lpo
                B_raw[t, :, 0], B_raw[t, :, 1] = zr, br
                r, term, trunc, info = env.step(a, z=z, beta=beta)
            elif outer_p is not None:
                z, beta = outer_p.current()
                r, term, trunc, info = env.step(a, z=z, beta=beta)
            else:
                r, term, trunc, info = env.step(a)
            ep_viable += 1.0 - info["viol"].mean(1)
            nobs = env.obs()
            r_used = r.astype(np.float64).copy()
            if norm_reward:
                ret_acc = ret_acc * gamma + r_used
                ret_rms.update(ret_acc)
                r_used = np.clip(r_used / math.sqrt(ret_rms.var + 1e-8), -10.0, 10.0)
            if trunc.any():
                r_used[trunc] += gamma * ag.np_value(nobs[trunc])
            B_o[t], B_a[t], B_lp[t], B_v[t], B_r[t] = obs, a, lp, v, r_used
            done = term | trunc
            B_d[t] = done
            ep_len += 1
            ep_ret += r
            steps += E
            for i in np.flatnonzero(done):
                if outer_p is not None:
                    # external fitness: survival (termination) or viable fraction (no termination)
                    fit = ep_len[i] / env.max_steps if env.terminate else ep_viable[i] / ep_len[i]
                    outer_p.episode_end(i, fit)
                ep_viable[i] = 0.0
                if norm_reward:
                    ret_acc[i] = 0.0
                log["ep_end_step"].append(steps)
                log["ep_len"].append(int(ep_len[i]))
                log["ep_ret"].append(float(ep_ret[i]))
                ep_len[i], ep_ret[i] = 0, 0.0
                env.reset_one(i, int(rng_ep.integers(1 << 31)))
            obs = env.obs() if done.any() else nobs
        # GAE
        nv = ag.np_value(obs)
        adv = np.zeros((n_steps, E), np.float32)
        gae = np.zeros(E)
        for t in reversed(range(n_steps)):
            mask = 1.0 - B_d[t]
            delta = B_r[t] + gamma * nv * mask - B_v[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            nv = B_v[t]
        ret = adv + B_v
        N = n_steps * E
        o = torch.as_tensor(B_o.reshape(N, -1))
        a_t = torch.as_tensor(B_a.reshape(N))
        raw = torch.as_tensor(B_raw.reshape(N, 2))
        olp = torch.as_tensor(B_lp.reshape(N))
        adv_t = torch.as_tensor(adv.reshape(N))
        ret_t = torch.as_tensor(ret.reshape(N))
        mb = N // n_mb
        # mean policy entropy on the rollout states, before this update (diagnostic only)
        with torch.no_grad():
            log["upd_entropy"].append([steps, float(Categorical(logits=ag.pi(o)).entropy().mean())])
        for _ in range(n_epochs):
            perm = torch.randperm(N)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                madv = adv_t[idx]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                nlp, ent, nv_ = ag.evaluate(o[idx], a_t[idx], raw[idx])
                ratio = torch.exp(nlp - olp[idx])
                pg = torch.max(-madv * ratio, -madv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                loss = pg + vf_coef * 0.5 * ((nv_ - ret_t[idx]) ** 2).mean() - ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ag.parameters(), max_grad_norm)
                opt.step()
        ag.sync()
        if outer_p is not None:
            outer_p.update()
        if (ag.learn_or or outer_p is not None) and (upd % 10 == 0 or upd == n_upd - 1):
            log["or_traj"].append([steps, ag.or_summary() if ag.learn_or else outer_p.summary()])
    log["outer_final"] = outer_p.summary() if outer_p is not None else None
    return log


def evaluate(ag: Agent, k, cond_kw, seeds, terminate=True, deterministic=False, act_seed=0):
    """Paired evaluation: one env per seed; policy sampled with a fixed action RNG
    (stochastic) or argmax (deterministic).  Adaptive OR is irrelevant at evaluation
    (the reward is not used by the policy)."""
    from forage_env import make_reward_fn
    env = ForageEnv(len(seeds), k, make_reward_fn("Alive"), terminate=terminate, **cond_kw)
    for i, s in enumerate(seeds):
        env.reset_one(i, s)
    rng = np.random.default_rng(act_seed)
    return rollout_eval(env, lambda e: ag.np_act(e.obs(), rng, deterministic)[0])


def eval_diagnostics(act_fn, ent_fn, k, cond_kw, seeds, terminate=True, gamma=0.99):
    """Diagnostics on the same paired episodes as `evaluate` (added 2026-11-20, second review).

    act_fn(env) -> actions; ent_fn(env) -> per-env policy entropy (nats) or None.  When act_fn
    uses the same action RNG as `evaluate`, the trajectories are identical (entropy does not
    consume random numbers).  Per episode, with D_t = mean_i v_{i,t}^2 (the drive of DQ):
      D0, DL      drive at the start and at the last step (at death, with termination);
      Dbar_disc   the discounted mean of the drives visited at steps 1..L-1, weights
                  (1-gamma) gamma^(t-1) renormalised to sum to 1 (NaN if L = 1);
      Dbar        the plain mean of D_1..D_{L-1};
      w_term      gamma^(L-1), the weight of D_L in the DQ return identity
                  sum_t gamma^t (D_t - D_{t+1}) = D_0 - [(1-gamma) sum_{t=1}^{L-1} gamma^(t-1) D_t
                  + gamma^(L-1) D_L];
      ret_dq      the discounted DQ return of the episode (checks the identity);
      entropy     mean policy entropy over the steps of the episode (nats)."""
    from forage_env import make_reward_fn
    env = ForageEnv(len(seeds), k, make_reward_fn("Alive"), terminate=terminate, **cond_kw)
    for i, s in enumerate(seeds):
        env.reset_one(i, s)
    E = env.E
    alive = np.ones(E, bool)
    L = np.zeros(E, np.int64)
    D0 = (env.v ** 2).mean(1)
    Dprev = D0.copy()
    DL = D0.copy()
    s_disc, w_disc, s_plain = np.zeros(E), np.zeros(E), np.zeros(E)
    ret = np.zeros(E)
    ent_sum = np.zeros(E)
    died = np.zeros(E, bool)
    while alive.any():
        if ent_fn is not None:
            ent_sum += alive * ent_fn(env)
        a = act_fn(env)
        _, term, trunc, info = env.step(a)
        D = (info["dev"] ** 2).mean(1)
        al = alive.copy()
        # D_prev (= D_{t}) enters the visited mean only if t >= 1, i.e. L >= 1 before this step
        add = al & (L >= 1)
        wt = (1 - gamma) * gamma ** np.maximum(L - 1, 0)
        s_disc += add * wt * Dprev
        w_disc += add * wt
        s_plain += add * Dprev
        ret += al * gamma ** L * (Dprev - D)
        L += al
        DL = np.where(al, D, DL)
        Dprev = np.where(al, D, Dprev)
        died |= al & term
        alive &= ~(term | trunc)
    out = []
    for i in range(E):
        n = int(L[i])
        out.append({"length": n, "event": bool(died[i]), "D0": float(D0[i]), "DL": float(DL[i]),
                    "Dbar_disc": float(s_disc[i] / w_disc[i]) if w_disc[i] > 0 else float("nan"),
                    "Dbar": float(s_plain[i] / (n - 1)) if n > 1 else float("nan"),
                    "w_term": float(gamma ** (n - 1)), "ret_dq": float(ret[i]),
                    "entropy": float(ent_sum[i] / max(n, 1)) if ent_fn is not None else None})
    return out


def agent_fns(ag: Agent, act_seed=0, deterministic=False):
    rng = np.random.default_rng(act_seed)

    def act(env):
        return ag.np_act(env.obs(), rng, deterministic)[0]

    def ent(env):
        lg = ag.np_logits(env.obs())
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(1, keepdims=True)
        return -(p * np.log(p + 1e-12)).sum(1)
    return act, ent


def recovery_test(ag: Agent, k, cond_kw, seeds, deficit=-0.8, zone=0.5, horizon=150, act_seed=0):
    """Homeostatic recovery test (criteria fixed before running, 2026-11-10).

    For each disturbed variable j and each episode seed: all variables start at 0 except
    v_j = `deficit`; the policy is run (stochastic, fixed action RNG) for at most `horizon`
    steps with termination enabled.  Success = v_j returns to |v_j| <= `zone` before the
    horizon and before any viability violation.  Recovery time is defined for successes only.
    """
    out = []
    for jvar in range(k):
        env = ForageEnv(len(seeds), k, lambda d0, d1, z=None, beta=None: np.zeros(d0.shape[0]),
                        terminate=True, **cond_kw)
        for i, s in enumerate(seeds):
            env.reset_one(i, s)
            env.v[i] = 0.0
            env.v[i, jvar] = deficit
            env.obs_v[i] = env._noisy(i)
        rng = np.random.default_rng(act_seed)
        E = env.E
        done = np.zeros(E, bool)
        success = np.zeros(E, bool)
        t_rec = np.full(E, -1)
        died = np.zeros(E, bool)
        for t in range(1, horizon + 1):
            a = ag.np_act(env.obs(), rng)[0]
            _, term, _, info = env.step(a)
            rec = (~done) & (np.abs(env.v[:, jvar]) <= zone) & (~term)
            success |= rec
            t_rec[rec] = t
            newly_dead = (~done) & term
            died |= newly_dead
            done |= rec | term
            if done.all():
                break
        for i, s in enumerate(seeds):
            out.append({"var": jvar, "seed": int(s), "success": bool(success[i]),
                        "t_rec": int(t_rec[i]) if success[i] else None, "died": bool(died[i])})
    return out
