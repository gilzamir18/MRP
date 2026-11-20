"""
ppo_agent.py — single PPO-Clip implementation used for EVERY MRP variant.

In the original pipeline Adaptive OR was trained with Stable-Baselines3 (actions
clipped to [0,1], z/beta emitted as state-dependent action dimensions) while all
other variants used a separate custom PPO.  Here every variant shares the same
code path; Adaptive OR differs only by the four state-independent scalars
theta = {mu_sz, log sigma_sz, mu_beta, log sigma_beta} (paper Eqs. 18-22):

    z~ ~ N(mu_sz, sigma_sz^2),   beta~ ~ N(mu_beta, sigma_beta^2)   (unconstrained)
    z    = sigmoid(z~) * 0.8 + 0.1          in (0.1, 0.9)
    beta = softplus(beta~)                  in (0, inf)
    log pi(a, z~, beta~ | s) = log pi(a|s) + log N(z~) + log N(beta~)

Defaults follow SB3 PPO / paper Table 2: separate 2x64 tanh MLPs for actor and
critic, orthogonal init, state-independent log-std (init 0), Adam(lr=3e-4,
eps=1e-5), n_steps=1024, batch 64, 10 epochs, gamma=0.99, GAE 0.95, clip 0.2,
vf_coef 0.5, ent_coef 0, max_grad_norm 0.5, per-minibatch advantage
normalisation, no value clipping.  Time-limit truncation is bootstrapped.
Optional SB3-VecNormalize-style reward normalisation (running std of the
discounted return).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

Normal.set_default_validate_args(False)
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def softplus_inv(y: float) -> float:
    return float(math.log(math.expm1(y))) if y < 20 else float(y)


def logit(p: float) -> float:
    return float(math.log(p / (1.0 - p)))


def z_from_raw(x):
    return torch.sigmoid(x) * 0.8 + 0.1


def beta_from_raw(x):
    return nn.functional.softplus(x)


def _mlp(inp, hid, out, out_gain, hid2=None):
    hid2 = hid if hid2 is None else hid2
    net = nn.Sequential(nn.Linear(inp, hid), nn.Tanh(), nn.Linear(hid, hid2), nn.Tanh(),
                        nn.Linear(hid2, out))
    for i, layer in enumerate(net):
        if isinstance(layer, nn.Linear):
            gain = out_gain if i == len(net) - 1 else math.sqrt(2)
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
    return net


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, learn_or=False,
                 z_init=0.5, beta_init=1.0, log_sigma_init=-1.0, log_std_init=0.0, arch="mlp"):
        super().__init__()
        self.act_dim = act_dim
        self.learn_or = learn_or
        self.arch = arch
        self.obs_rms = None          # set by train(norm_obs=True); frozen at evaluation
        # "mlp": joint MLP over the whole observation (paper / SB3 default)
        # "shared": permutation-equivariant actor; the same small MLP maps each variable's
        #           features (v_i, |v_i|, 100*decay_i, budget, mean|v|) to its logit.
        # "wide": joint MLP over all variables sized by the input: k features per variable
        #         (k*n inputs + the global budget), two hidden layers of 2*(k*n)+1 units each.
        if arch == "wide":
            k = (obs_dim - 1) // act_dim
            hidden = 2 * k * act_dim + 1
        hidden2 = None
        if arch == "wide2":
            # joint MLP sized by the input size m = 4n+1: hidden layers 2m+1 and m
            hidden, hidden2 = 2 * obs_dim + 1, obs_dim
        self.hidden = hidden
        self.pi = _mlp(self.N_FEAT if arch == "shared" else obs_dim, hidden, 1 if arch == "shared" else act_dim, 0.01, hidden2)
        self.vf = _mlp(obs_dim, hidden, 1, 1.0, hidden2)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        if learn_or:
            # mu_sz such that sigmoid(mu_sz)*0.8+0.1 = z_init ; softplus(mu_beta) = beta_init
            self.mu_sz = nn.Parameter(torch.tensor([logit((z_init - 0.1) / 0.8)]))
            self.log_sig_sz = nn.Parameter(torch.tensor([log_sigma_init]))
            self.mu_beta = nn.Parameter(torch.tensor([softplus_inv(beta_init)]))
            self.log_sig_beta = nn.Parameter(torch.tensor([log_sigma_init]))

    # ------------------------------------------------------------------
    def value(self, obs):
        return self.vf(obs).squeeze(-1)

    N_FEAT = 5

    def _feat_np(self, x):
        n = self.act_dim
        v, dec, b = x[:n], x[3 * n:4 * n], x[4 * n]
        av = np.abs(v)
        return np.stack([v, av, 100.0 * dec, np.full(n, b), np.full(n, av.mean())], axis=1)

    def _feat_t(self, x):
        n = self.act_dim
        v, dec, b = x[:, :n], x[:, 3 * n:4 * n], x[:, 4 * n:4 * n + 1]
        av = v.abs()
        return torch.stack([v, av, 100.0 * dec, b.expand_as(v), av.mean(1, keepdim=True).expand_as(v)], dim=2)

    def policy_mean(self, obs):
        if self.arch == "shared":
            return self.pi(self._feat_t(obs)).squeeze(-1)
        return self.pi(obs)

    def _dists(self, obs):
        mean = self.policy_mean(obs)
        d_ctrl = Normal(mean, self.log_std.exp().expand_as(mean))
        if not self.learn_or:
            return d_ctrl, None, None
        d_z = Normal(self.mu_sz, self.log_sig_sz.exp())
        d_b = Normal(self.mu_beta, self.log_sig_beta.exp())
        return d_ctrl, d_z, d_b

    # ---- fast numpy inference for rollouts (weights synced after each update) ----
    def sync_numpy(self, rng=None):
        with torch.no_grad():
            self._np = {
                "pi": [(l.weight.numpy().astype(np.float64).T.copy(), l.bias.numpy().astype(np.float64).copy())
                       for l in self.pi if isinstance(l, nn.Linear)],
                "vf": [(l.weight.numpy().astype(np.float64).T.copy(), l.bias.numpy().astype(np.float64).copy())
                       for l in self.vf if isinstance(l, nn.Linear)],
                "std": self.log_std.exp().numpy().astype(np.float64).copy(),
                "log_std": self.log_std.numpy().astype(np.float64).copy(),
            }
            if self.learn_or:
                self._np["or"] = (float(self.mu_sz), float(self.log_sig_sz),
                                  float(self.mu_beta), float(self.log_sig_beta))
        if rng is not None:
            self._rng = rng
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(0)

    @staticmethod
    def _np_mlp(layers, x):
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < len(layers) - 1:
                x = np.tanh(x)
        return x

    def prep(self, obs_np):
        """Observation normalisation (VecNormalize-style); identity when disabled."""
        x = np.asarray(obs_np, np.float64)
        if self.obs_rms is None:
            return x
        return np.clip((x - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0)

    def np_value(self, obs_np):
        return float(self._np_mlp(self._np["vf"], self.prep(obs_np))[0])

    def act(self, obs_np, deterministic=False):
        if not hasattr(self, "_np"):
            self.sync_numpy()
        P = self._np
        x = self.prep(obs_np)
        if self.arch == "shared":
            mean = self._np_mlp(P["pi"], self._feat_np(x))[:, 0]
        else:
            mean = self._np_mlp(P["pi"], x)
        if deterministic:
            a = mean
        else:
            a = mean + P["std"] * self._rng.standard_normal(mean.shape)
        lp = float(np.sum(-0.5 * ((a - mean) / P["std"]) ** 2 - P["log_std"] - _HALF_LOG_2PI))
        extra = None
        if self.learn_or:
            mz, lsz, mb, lsb = P["or"]
            if deterministic:
                zr, br = mz, mb
            else:
                zr = mz + math.exp(lsz) * float(self._rng.standard_normal())
                br = mb + math.exp(lsb) * float(self._rng.standard_normal())
            lp += (-0.5 * ((zr - mz) / math.exp(lsz)) ** 2 - lsz - _HALF_LOG_2PI
                   - 0.5 * ((br - mb) / math.exp(lsb)) ** 2 - lsb - _HALF_LOG_2PI)
            z = 1.0 / (1.0 + math.exp(-zr)) * 0.8 + 0.1
            beta = math.log1p(math.exp(br)) if br < 30 else br
            extra = (zr, br, z, beta)
        v = float(self._np_mlp(P["vf"], x)[0])
        return a.astype(np.float32), lp, v, extra

    def evaluate(self, obs, act, raw_params):
        d_ctrl, d_z, d_b = self._dists(obs)
        lp = d_ctrl.log_prob(act).sum(-1)
        ent = d_ctrl.entropy().sum(-1)
        if self.learn_or:
            lp = lp + d_z.log_prob(raw_params[:, 0:1]).sum(-1) + d_b.log_prob(raw_params[:, 1:2]).sum(-1)
            ent = ent + d_z.entropy().sum() + d_b.entropy().sum()
        return lp, ent, self.value(obs)

    def or_params_summary(self):
        """Raw and transformed values of the learned OR parameters."""
        if not self.learn_or:
            return None
        with torch.no_grad():
            mu_sz, s_sz = float(self.mu_sz), float(self.log_sig_sz.exp())
            mu_b, s_b = float(self.mu_beta), float(self.log_sig_beta.exp())
            # expectation of the transformed samples (Monte-Carlo)
            g = torch.Generator().manual_seed(0)
            eps = torch.randn(20000, generator=g)
            ez = float(z_from_raw(mu_sz + s_sz * eps).mean())
            eb = float(beta_from_raw(mu_b + s_b * eps).mean())
        return {
            "mu_sz_raw": mu_sz, "sigma_sz": s_sz, "mu_beta_raw": mu_b, "sigma_beta": s_b,
            "z_of_mean": float(z_from_raw(torch.tensor(mu_sz))),       # transform of the mean (Eq. 18)
            "beta_of_mean": float(beta_from_raw(torch.tensor(mu_b))),  # transform of the mean (Eq. 19)
            "z_mean_of_samples": ez, "beta_mean_of_samples": eb,
        }


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


class VecRunningMeanStd:
    """Per-feature running mean/variance (as in SB3 VecNormalize)."""
    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, len(self.mean))
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


def train(env, ac: ActorCritic, total_timesteps=100_000, n_steps=1024, batch_size=64,
          n_epochs=10, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, vf_coef=0.5,
          ent_coef=0.0, max_grad_norm=0.5, norm_reward=False, episode_seed_rng=None,
          action_rng=None, norm_obs=False, checkpoints=None, on_checkpoint=None):
    """Train in place.  Returns a log dict (training curves + OR parameter trajectory).

    checkpoints / on_checkpoint (added 2026-11-20, second review): rollouts are cut at each
    checkpoint step so that the agent at checkpoint c is exactly the agent a run with
    total_timesteps = c would return; on_checkpoint(c) is called after that update.  The
    callback must not consume the training RNGs (deterministic evaluation does not)."""
    cps = sorted(c for c in (checkpoints or []) if 0 < c < total_timesteps)
    opt = torch.optim.Adam(ac.parameters(), lr=lr, eps=1e-5, foreach=True)
    if norm_obs:
        ac.obs_rms = VecRunningMeanStd(env.obs_dim)
    ac.sync_numpy(action_rng or np.random.default_rng(1))
    rng = episode_seed_rng or np.random.default_rng(0)
    ret_rms, ret_acc = RunningMeanStd(), 0.0

    obs = env.reset(int(rng.integers(1 << 31)))
    ep_len, ep_ret = 0, 0.0
    log = {"ep_end_step": [], "ep_len": [], "ep_ret": [], "upd_step": [], "or_params": []}
    steps = 0
    n_p = 2 if ac.learn_or else 0

    while steps < total_timesteps:
        T = min(n_steps, total_timesteps - steps)
        if cps:
            T = min(T, cps[0] - steps)
        B_obs = np.zeros((T, env.obs_dim), np.float32)
        B_act = np.zeros((T, env.act_dim), np.float32)
        B_raw = np.zeros((T, max(n_p, 1)), np.float32)
        B_lp = np.zeros(T, np.float32)
        B_val = np.zeros(T, np.float32)
        B_rew = np.zeros(T, np.float32)
        B_done = np.zeros(T, np.float32)

        for t in range(T):
            if norm_obs:
                ac.obs_rms.update(obs)
            a, lp, v, extra = ac.act(obs)
            if extra is not None:
                zr, br, z, beta = extra
                B_raw[t] = (zr, br)
                nobs, r, term, trunc, _ = env.step(a, z=z, beta=beta)
            else:
                nobs, r, term, trunc, _ = env.step(a)
            ep_len += 1
            ep_ret += r
            if norm_reward:
                ret_acc = ret_acc * gamma + r
                ret_rms.update([ret_acc])
                r_used = float(np.clip(r / math.sqrt(ret_rms.var + 1e-8), -10, 10))
            else:
                r_used = r
            if trunc:  # bootstrap time-limit truncation
                r_used += gamma * ac.np_value(nobs)
            B_obs[t], B_act[t], B_lp[t], B_val[t], B_rew[t] = ac.prep(obs), a, lp, v, r_used
            B_done[t] = float(term or trunc)
            steps += 1
            if term or trunc:
                log["ep_end_step"].append(steps)
                log["ep_len"].append(ep_len)
                log["ep_ret"].append(ep_ret)
                ep_len, ep_ret, ret_acc = 0, 0.0, 0.0
                obs = env.reset(int(rng.integers(1 << 31)))
            else:
                obs = nobs

        last_v = ac.np_value(obs)
        adv = np.zeros(T, np.float32)
        gae, nv = 0.0, last_v
        for t in reversed(range(T)):
            mask = 1.0 - B_done[t]
            delta = B_rew[t] + gamma * nv * mask - B_val[t]
            gae = delta + gamma * gae_lambda * mask * gae
            adv[t] = gae
            nv = B_val[t]
        ret = adv + B_val

        o = torch.as_tensor(B_obs)
        ac_t = torch.as_tensor(B_act)
        raw_t = torch.as_tensor(B_raw)
        olp = torch.as_tensor(B_lp)
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret)
        for _ in range(n_epochs):
            perm = torch.randperm(T)
            for s in range(0, T, batch_size):
                idx = perm[s:s + batch_size]
                mb_adv = adv_t[idx]
                if len(idx) > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                nlp, ent, nv_ = ac.evaluate(o[idx], ac_t[idx], raw_t[idx])
                ratio = torch.exp(nlp - olp[idx])
                pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                vl = 0.5 * ((nv_ - ret_t[idx]) ** 2).mean()
                loss = pg + vf_coef * vl - ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                opt.step()

        ac.sync_numpy()
        log["upd_step"].append(steps)
        log["or_params"].append(ac.or_params_summary())
        if cps and steps >= cps[0]:
            c = cps.pop(0)
            if on_checkpoint is not None:
                on_checkpoint(c)
    return log


def evaluate(env, ac: ActorCritic, seeds, ref_zone=0.5):
    """Deterministic evaluation on a fixed list of episode seeds.

    Returns per-episode length, whether it ended by termination (death) or by
    the horizon (censored), and behavioural diagnostics.
    """
    out = []
    z = beta = None
    if ac.learn_or:
        p = ac.or_params_summary()
        z, beta = p["z_of_mean"], p["beta_of_mean"]
    for sd in seeds:
        obs = env.reset(int(sd))
        done, L = False, 0
        in_zone, alloc_ent, alloc_corr, top_share, n_nonrec = [], [], [], [], []
        viable, all_viable, mean_dev = [], [], []
        first_viol = None
        dead_var_init = dead_var_decay_pct = None
        v0 = np.abs(env.v.copy())
        while not done:
            a, _, _, _ = ac.act(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a, z=z, beta=beta)
            L += 1
            al, dv, d0 = info["alloc"], info["dev"], info["dev_prev"]
            p_ = al / al.sum()
            alloc_ent.append(float(-(p_ * np.log(p_ + 1e-12)).sum() / np.log(len(p_))))
            if dv.std() > 0 and al.std() > 0:
                alloc_corr.append(float(np.corrcoef(al, d0)[0, 1]))
            k = max(1, len(dv) // 10)
            top = np.argsort(-d0)[:k]
            top_share.append(float(p_[top].sum()))
            in_zone.append(float((dv <= ref_zone).mean()))
            n_nonrec.append(float(((dv > ref_zone) & (dv >= d0)).mean()))
            vi = info["viol"]
            viable.append(1.0 - float(vi.mean()))
            all_viable.append(float(not vi.any()))
            mean_dev.append(float(dv.mean()))
            if first_viol is None and vi.any():
                first_viol = L
            done = term or trunc
            if term:
                j = int(np.argmax(dv))
                dead_var_init = float(v0[j])
                dead_var_decay_pct = float((env.decay < env.decay[j]).mean())
        out.append({
            "seed": int(sd), "length": L, "event": bool(term),
            "frac_in_zone": float(np.mean(in_zone)),
            "frac_nonrecovery": float(np.mean(n_nonrec)),
            "alloc_entropy": float(np.mean(alloc_ent)),
            "alloc_dev_corr": float(np.mean(alloc_corr)) if alloc_corr else float("nan"),
            "top10_share": float(np.mean(top_share)),
            "dead_var_init_dev": dead_var_init, "dead_var_decay_pct": dead_var_decay_pct,
            # viability diagnostics (meaningful in the no-termination control)
            "frac_viable": float(np.mean(viable)), "frac_all_viable": float(np.mean(all_viable)),
            "mean_dev": float(np.mean(mean_dev)),
            "first_viol": first_viol if first_viol is not None else L,
            "viol_event": first_viol is not None,
        })
    return out
