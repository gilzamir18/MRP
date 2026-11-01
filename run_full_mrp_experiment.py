"""
run_full_mrp_experiment.py
===========================
Orquestrador dos experimentos do Multi-Resource Problem (MRP) e gerador de:
  1. Tabelas Estatísticas (Tabela 6: Duração Média de Episódio, Tabela 8: Parâmetros Aprendidos).
  2. Testes de Hipótese Não-Paramétricos (Kruskal-Wallis H, Mann-Whitney U com correção de Bonferroni e tamanho de efeito r de correlação ponto-bisserial).
  3. Gráficos completos:
     - Curvas de aprendizado ao longo dos timesteps para as dimensões configuradas (ex: n=50 e n=100) [Figuras 4 e 5].
     - Gráfico de barras comparativo de desempenho entre OR, QD e Default para C1-C8.
     - Gráfico da evolução e adaptação dos parâmetros aprendidos z_safe e beta por regime de orçamento.
"""

import argparse
import sys
import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_palette("muted")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from env import HomeostaticEnv

VARIANTS = ["OR", "QD", "default"]
COLORS = {"OR": "#e05c5c", "QD": "#4c8ef5", "default": "#4cba6b"}
CURVE_POINTS = 200


def build_conditions(dims=(50, 100)):
    n1, n2 = dims[0], dims[1]
    return {
        "C1": {"n": n1, "obs_noise": 0.0, "budget": 1.00, "name": f"n={n1}, no noise, b=1.0"},
        "C2": {"n": n1, "obs_noise": 0.2, "budget": 1.00, "name": f"n={n1}, noise=0.2, b=1.0"},
        "C3": {"n": n1, "obs_noise": 0.0, "budget": 0.10, "name": f"n={n1}, no noise, b=0.1"},
        "C4": {"n": n1, "obs_noise": 0.2, "budget": 0.10, "name": f"n={n1}, noise=0.2, b=0.1"},
        "C5": {"n": n2, "obs_noise": 0.0, "budget": 1.00, "name": f"n={n2}, no noise, b=1.0"},
        "C6": {"n": n2, "obs_noise": 0.2, "budget": 1.00, "name": f"n={n2}, noise=0.2, b=1.0"},
        "C7": {"n": n2, "obs_noise": 0.0, "budget": 0.10, "name": f"n={n2}, no noise, b=0.1"},
        "C8": {"n": n2, "obs_noise": 0.2, "budget": 0.10, "name": f"n={n2}, noise=0.2, b=0.1"},
    }


# ── WRAPPER DE PARÂMETROS APRENDIDOS PELO OR (PPO-CLIP) ───────────────────────
class ORParamWrapper(gym.Wrapper):
    """Wrapper que estende o espaço de ações com os parâmetros logit de z_safe e beta."""

    def __init__(self, env: HomeostaticEnv):
        super().__init__(env)
        ctrl_dim = env.action_space.shape[0]
        low  = np.concatenate([env.action_space.low,  [-5.0, -5.0]], dtype=np.float32)
        high = np.concatenate([env.action_space.high, [ 5.0,  5.0]], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._ctrl_dim = ctrl_dim

    @staticmethod
    def decode_sz(sz_raw: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(sz_raw)))) * 0.8 + 0.1

    @staticmethod
    def decode_beta(beta_raw: float) -> float:
        x = float(beta_raw)
        return float(np.log1p(np.exp(x))) if x < 30 else x

    def step(self, action):
        ctrl = action[:self._ctrl_dim]
        sz   = self.decode_sz(action[self._ctrl_dim])
        beta = self.decode_beta(action[self._ctrl_dim + 1])
        return self.env.step(ctrl, learned_params={"safety_zone": sz, "penalty_ratio": beta})


# ── REDE NEURAL ACTOR-CRITIC PARA BASELINES QD E DEFAULT ──────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.log_std     = nn.Parameter(torch.zeros(action_dim))
        self._init_weights()

    def _init_weights(self):
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        mean  = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        std   = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self(obs)
        dist     = Normal(mean, std)
        action   = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self(obs)
        dist     = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def _compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    advantages, gae = [], 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        mask  = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae   = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


class _RolloutBuffer:
    def __init__(self):
        self.obs = []; self.actions = []; self.log_probs = []
        self.values = []; self.rewards = []; self.dones = []

    def clear(self): self.__init__()

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs); self.actions.append(action)
        self.log_probs.append(log_prob); self.values.append(value)
        self.rewards.append(reward); self.dones.append(done)

    def to_tensors(self, advantages, returns, device="cpu"):
        return (
            torch.FloatTensor(np.array(self.obs)).to(device),
            torch.FloatTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(self.log_probs).to(device),
            torch.FloatTensor(advantages).to(device),
            torch.FloatTensor(returns).to(device),
        )


def _ppo_update(ac, optimizer, obs_t, actions_t, old_lp_t, adv_t, ret_t, old_val_t,
                n_epochs, batch_size, clip_range, vf_coef, ent_coef, max_grad_norm):
    n = obs_t.shape[0]
    for _ in range(n_epochs):
        for start in range(0, n, batch_size):
            idx = torch.randperm(n)[start: start + batch_size]
            mb_adv = adv_t[idx]
            if mb_adv.shape[0] > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            new_lp, entropy, new_val = ac.evaluate(obs_t[idx], actions_t[idx])
            ratio   = torch.exp(new_lp - old_lp_t[idx])
            pg_loss = torch.max(-mb_adv * ratio,
                                -mb_adv * torch.clamp(ratio, 1-clip_range, 1+clip_range)).mean()

            vf_unclipped = (new_val - ret_t[idx]) ** 2
            vf_clipped   = (old_val_t[idx] + torch.clamp(new_val - old_val_t[idx],
                             -clip_range, clip_range) - ret_t[idx]) ** 2
            vf_loss = 0.5 * torch.max(vf_unclipped, vf_clipped).mean()

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            optimizer.step()


def _train_ppo_custom(env, ac, optimizer, total_timesteps, n_steps, n_epochs, batch_size,
                       gamma, gae_lambda, clip_range, vf_coef, ent_coef, max_grad_norm, device):
    buffer = _RolloutBuffer()
    obs, _ = env.reset()
    done = truncated = False
    total_steps = 0
    ep_len = 0
    ep_curve = []

    pbar = tqdm(total=total_timesteps, desc="  treino", leave=False, unit="step")

    while total_steps < total_timesteps:
        buffer.clear()
        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = ac.get_action(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, _ = env.step(action_np)
            buffer.add(obs, action_np, log_prob.item(), value.item(),
                       float(reward), bool(done or truncated))
            total_steps += 1
            ep_len += 1
            pbar.update(1)

            if done or truncated:
                ep_curve.append((total_steps, ep_len))
                ep_len = 0
                obs, _ = env.reset()
                done = truncated = False
            else:
                obs = next_obs

            if total_steps >= total_timesteps:
                break

        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            _, _, last_v = ac.get_action(last_obs_t)
            last_value = last_v.item() if not (done or truncated) else 0.0

        advantages, returns = _compute_gae(
            buffer.rewards, buffer.values, buffer.dones, last_value, gamma, gae_lambda
        )
        obs_t, act_t, lp_t, adv_t, ret_t = buffer.to_tensors(advantages, returns, device)
        old_val_t = torch.FloatTensor(buffer.values).to(device)

        _ppo_update(ac, optimizer, obs_t, act_t, lp_t, adv_t, ret_t, old_val_t,
                    n_epochs, batch_size, clip_range, vf_coef, ent_coef, max_grad_norm)

    pbar.close()
    return ep_curve


class EpisodeCurveCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self._total = total_timesteps
        self._pbar = None
        self.ep_curve = []
        self._ep_len = 0

    def _on_training_start(self) -> None:
        self._pbar = tqdm(total=self._total, desc="  treino", leave=False, unit="step")

    def _on_step(self) -> bool:
        n = self.training_env.num_envs
        self._pbar.update(n)
        self._ep_len += n
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self.ep_curve.append((self.num_timesteps, self._ep_len))
                self._ep_len = 0
        return True

    def _on_training_end(self) -> None:
        self._pbar.close()


def interpolate_curve(ep_curve, total_timesteps, n_points=CURVE_POINTS, smooth_window=10):
    if not ep_curve:
        xs = np.linspace(0, total_timesteps, n_points)
        return xs, np.zeros(n_points)

    ts = np.array([t for t, _ in ep_curve], dtype=float)
    ys = np.array([l for _, l in ep_curve], dtype=float)

    if len(ys) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        ys_sm  = np.convolve(ys, kernel, mode="same")
        for i in range(smooth_window // 2):
            ys_sm[i]      = ys[:i+1].mean()
            ys_sm[-(i+1)] = ys[-(i+1):].mean()
    else:
        ys_sm = ys.copy()

    xs_grid = np.linspace(ts[0], ts[-1], n_points)
    ys_grid = np.interp(xs_grid, ts, ys_sm)

    xs_full = np.linspace(0, total_timesteps, n_points)
    ys_full = np.zeros(n_points)
    idx_start = np.searchsorted(xs_full, xs_grid[0])
    idx_end   = np.searchsorted(xs_full, xs_grid[-1])
    if idx_end >= n_points:
        idx_end = n_points - 1
    n_fill = idx_end - idx_start + 1
    ys_full[idx_start:idx_end+1] = np.interp(
        np.linspace(xs_grid[0], xs_grid[-1], n_fill), xs_grid, ys_grid
    )
    ys_full[:idx_start] = ys_full[idx_start]
    ys_full[idx_end+1:] = ys_full[idx_end]

    return xs_full, ys_full


# ── EXECUÇÃO DE EXPERIMENTO INDIVIDUAL ─────────────────────────────────────────
def run_single_experiment(
    variant: str, train_seed: int, cond_params: dict,
    total_timesteps: int, n_eval_episodes: int, eval_seed_start: int,
    learning_rate: float, n_steps: int, batch_size: int, n_epochs: int,
    hidden_dim: int, device: str
) -> dict:
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    import random
    random.seed(train_seed)

    num_vars = cond_params["n"]
    learn_or = (variant == "OR")
    reward_scale = 1.0 / num_vars

    env_kwargs = dict(
        num_vars           = num_vars,
        rewarding          = variant,
        reward_scale       = reward_scale,
        max_steps          = 1000,
        safety_zone        = 0.5,
        penalty_ratio      = 3.0,
        obs_noise_std      = cond_params["obs_noise"],
        decay_std          = 0.001,
        decay_ratio_mu     = 0.001,
        correction_budget  = cond_params["budget"],
        use_learned_params = learn_or,
    )

    ep_curve = []

    if learn_or:
        base_env  = HomeostaticEnv(**env_kwargs)
        base_env.reset(seed=train_seed)
        train_env = ORParamWrapper(base_env)
        cb = EpisodeCurveCallback(total_timesteps)
        model = PPO(
            policy        = "MlpPolicy",
            env           = train_env,
            learning_rate = learning_rate,
            n_steps       = n_steps,
            batch_size    = batch_size,
            n_epochs      = n_epochs,
            policy_kwargs = {"net_arch": [hidden_dim, hidden_dim]},
            seed          = train_seed,
            device        = device,
            verbose       = 0,
        )
        model.learn(total_timesteps=total_timesteps, callback=cb)
        ep_curve = cb.ep_curve
        train_env.close()

    else:
        train_env = HomeostaticEnv(**env_kwargs)
        train_env.reset(seed=train_seed)
        obs_dim    = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        ac         = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        optimizer  = optim.Adam(ac.parameters(), lr=learning_rate, eps=1e-5)

        ep_curve = _train_ppo_custom(
            train_env, ac, optimizer,
            total_timesteps = total_timesteps,
            n_steps         = n_steps,
            n_epochs        = n_epochs,
            batch_size      = batch_size,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            vf_coef         = 0.5,
            ent_coef        = 0.0,
            max_grad_norm   = 0.5,
            device          = device,
        )
        train_env.close()
        ac.eval()

    base_eval = HomeostaticEnv(**env_kwargs)
    eval_env  = ORParamWrapper(base_eval) if learn_or else base_eval

    ep_lengths   = []
    sz_samples   = []
    beta_samples = []

    for ep_i in range(n_eval_episodes):
        obs, _ = eval_env.reset(seed=eval_seed_start + ep_i)
        ep_len = 0
        done = truncated = False
        while not (done or truncated):
            if learn_or:
                action, _ = model.predict(obs, deterministic=True)
                sz_samples.append(ORParamWrapper.decode_sz(action[num_vars]))
                beta_samples.append(ORParamWrapper.decode_beta(action[num_vars + 1]))
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action_t, _, _ = ac.get_action(obs_t, deterministic=True)
                    action = action_t.squeeze(0).cpu().numpy()
            obs, _, done, truncated, _ = eval_env.step(action)
            ep_len += 1
        ep_lengths.append(ep_len)

    eval_env.close()

    result = {
        "variant"       : variant,
        "train_seed"    : train_seed,
        "ep_curve"      : ep_curve,
        "ep_lengths"    : ep_lengths,
        "mean_ep_len"   : float(np.mean(ep_lengths)),
        "median_ep_len" : float(np.median(ep_lengths)),
    }

    if learn_or and sz_samples:
        result["or_sz_mean"]   = float(np.mean(sz_samples))
        result["or_sz_std"]    = float(np.std(sz_samples))
        result["or_beta_mean"] = float(np.mean(beta_samples))
        result["or_beta_std"]  = float(np.std(beta_samples))

    return result


def rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * u) / (n1 * n2)


# ── GERADORES DE GRÁFICOS COMPLETOS ───────────────────────────────────────────
def plot_learning_curves(all_results_by_cond, conditions_dict, total_timesteps, n_vars_target, filename):
    os.makedirs("figures", exist_ok=True)
    cond_keys = [k for k, v in conditions_dict.items() if v["n"] == n_vars_target]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    xs_common = np.linspace(0, total_timesteps, CURVE_POINTS)

    for idx, cond in enumerate(cond_keys):
        ax = axes[idx]
        cond_data = all_results_by_cond[cond]

        for variant in VARIANTS:
            runs = cond_data.get(variant, [])
            if not runs:
                continue
            curves = []
            for res in runs:
                _, ys = interpolate_curve(res["ep_curve"], total_timesteps)
                curves.append(ys)

            curves_arr = np.array(curves)
            mean_curve = curves_arr.mean(axis=0)
            std_curve  = curves_arr.std(axis=0)

            color = COLORS[variant]
            ax.plot(xs_common, mean_curve, label=variant, color=color, linewidth=2)
            ax.fill_between(xs_common, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.2, color=color)

        info = conditions_dict[cond]
        ax.set_title(f"{cond}: n={info['n']}, obs_noise={info['obs_noise']}, budget={info['budget']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Training timesteps", fontsize=10)
        ax.set_ylabel("Mean Episode Length", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Multi-Resource Problem (MRP) Learning Curves (n = {n_vars_target} variables)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join("figures", filename)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" -> Grafico salvo em: {output_path}")


def plot_summary_barplots(all_results_by_cond, conditions_dict):
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    cond_labels = list(conditions_dict.keys())
    x = np.arange(len(cond_labels))
    width = 0.25

    or_means, or_stds = [], []
    qd_means, qd_stds = [], []
    def_means, def_stds = [], []

    for cond in cond_labels:
        or_m = [r["mean_ep_len"] for r in all_results_by_cond[cond]["OR"]]
        qd_m = [r["mean_ep_len"] for r in all_results_by_cond[cond]["QD"]]
        df_m = [r["mean_ep_len"] for r in all_results_by_cond[cond]["default"]]

        or_means.append(np.mean(or_m)); or_stds.append(np.std(or_m))
        qd_means.append(np.mean(qd_m)); qd_stds.append(np.std(qd_m))
        def_means.append(np.mean(df_m)); def_stds.append(np.std(df_m))

    rects1 = ax.bar(x - width, or_means, width, yerr=or_stds, label='Operational Regimes (OR)', color=COLORS['OR'], capsize=4)
    rects2 = ax.bar(x, qd_means, width, yerr=qd_stds, label='Differential Quadratic (QD)', color=COLORS['QD'], capsize=4)
    rects3 = ax.bar(x + width, def_means, width, yerr=def_stds, label='Standard Homeostatic (DR)', color=COLORS['default'], capsize=4)

    ax.set_ylabel('Mean Episode Length', fontsize=12, fontweight="bold")
    ax.set_xlabel('MRP Experimental Condition (C1 - C8)', fontsize=12, fontweight="bold")
    ax.set_title('MRP Performance Comparison across Formulations (C1 - C8)', fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join("figures", "mrp_performance_comparison_bar.png")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" -> Grafico comparativo salvo em: {output_path}")


def plot_learned_parameters(all_results_by_cond, conditions_dict):
    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    conds = list(conditions_dict.keys())
    sz_means, sz_stds = [], []
    beta_means, beta_stds = [], []

    for cond in conds:
        runs = all_results_by_cond[cond]["OR"]
        sz_vals   = [r["or_sz_mean"]   for r in runs if "or_sz_mean" in r]
        beta_vals = [r["or_beta_mean"] for r in runs if "or_beta_mean" in r]

        sz_means.append(np.mean(sz_vals) if sz_vals else 0)
        sz_stds.append(np.std(sz_vals) if sz_vals else 0)
        beta_means.append(np.mean(beta_vals) if beta_vals else 0)
        beta_stds.append(np.std(beta_vals) if beta_vals else 0)

    x = np.arange(len(conds))

    ax1.bar(x, sz_means, yerr=sz_stds, color='#8884d8', capsize=4, alpha=0.85)
    ax1.set_title('Learned Safety Zone (z_safe) by Condition', fontsize=12, fontweight="bold")
    ax1.set_ylabel('z_safe (half-width)', fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(conds)
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x, beta_means, yerr=beta_stds, color='#82ca9d', capsize=4, alpha=0.85)
    ax2.set_title('Learned Asymmetric Penalty Ratio (beta) by Condition', fontsize=12, fontweight="bold")
    ax2.set_ylabel('beta (penalty ratio)', fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(conds)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle("Jointly Learned Regulatory Parameters in Operational Regimes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join("figures", "mrp_learned_parameters.png")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" -> Grafico de parametros aprendidos salvo em: {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Execucao dos Experimentos do MRP (C1-C8)")
    parser.add_argument("--seeds-per-cond", type=int, default=5, help="Numero de seeds de treino por combinacao")
    parser.add_argument("--eval-episodes",  type=int, default=30, help="Numero de episodios de avaliacao por seed")
    parser.add_argument("--timesteps",      type=int, default=300000, help="Timesteps por treino")
    parser.add_argument("--dims",           type=int, nargs=2, default=[50, 100], help="Dimensoes n1 e n2 (default: 50 100)")
    parser.add_argument("--device",         type=str, default="cpu", help="Dispositivo PyTorch ('cpu' ou 'cuda', default: cpu)")
    parser.add_argument("--quick-test",     action="store_true", help="Execucao ultra-rapida para verificacao")
    args = parser.parse_args()

    conditions_dict = build_conditions(args.dims)
    train_seeds = list(range(42, 42 + args.seeds_per_cond))
    eval_episodes = args.eval_episodes
    timesteps = args.timesteps

    if args.quick_test:
        print("[MODO QUICK TEST ATIVADO]")
        train_seeds = [42, 43]
        eval_episodes = 5
        timesteps = 20000

    print("=" * 70)
    print(f"  EXPERIMENTOS DO MULTI-RESOURCE PROBLEM (MRP)")
    print(f"  Dimensoes Configuraradas: {args.dims[0]} e {args.dims[1]} variaveis")
    print(f"  Seeds: {train_seeds} | Eval Ep: {eval_episodes} | Timesteps: {timesteps}")
    print(f"  Dispositivo PyTorch: {args.device}")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    all_results = {}

    for cond_id, cond_params in conditions_dict.items():
        print(f"\n>>> Executando Condicao {cond_id}: {cond_params['name']} ...")
        all_results[cond_id] = {v: [] for v in VARIANTS}

        for variant in VARIANTS:
            for seed in train_seeds:
                print(f"  [{cond_id}] {variant:<8} seed={seed} ... ", end="", flush=True)
                res = run_single_experiment(
                    variant=variant, train_seed=seed, cond_params=cond_params,
                    total_timesteps=timesteps, n_eval_episodes=eval_episodes,
                    eval_seed_start=1000, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, n_epochs=10, hidden_dim=64, device=args.device
                )
                all_results[cond_id][variant].append(res)
                print(f"-> mean={res['mean_ep_len']:.1f} median={res['median_ep_len']:.1f}", end="")
                if "or_sz_mean" in res:
                    print(f" | OR sz={res['or_sz_mean']:.3f} beta={res['or_beta_mean']:.3f}", end="")
                print()

    # ── TABELA 6: RESULTADOS DE DURAÇÃO DE EPISÓDIO (ESTATÍSTICAS C1-C8) ──────
    print("\n" + "=" * 80)
    print("TABELA 6: Duração Média de Episódio (µ) e Testes Estatísticos por Condição")
    print("=" * 80)
    print(f"{'Cond.':<6} {'n':<4} {'sigma':<6} {'b':<5} {'OR µ':<10} {'QD µ':<10} {'Def µ':<10} {'KW / Melhor':<25}")
    print("-" * 80)

    summary_rows = []
    pairwise_rows = []

    for cond_id, cond_params in conditions_dict.items():
        or_means  = [r["mean_ep_len"] for r in all_results[cond_id]["OR"]]
        qd_means  = [r["mean_ep_len"] for r in all_results[cond_id]["QD"]]
        def_means = [r["mean_ep_len"] for r in all_results[cond_id]["default"]]

        or_u   = np.mean(or_means)
        qd_u   = np.mean(qd_means)
        def_u  = np.mean(def_means)

        # Teste de Kruskal-Wallis Omnibus
        H, p_kw = scipy_stats.kruskal(or_means, qd_means, def_means)

        # Testes Par-a-Par Mann-Whitney U com Correção de Bonferroni (alpha_corr = 0.05 / 3 = 0.0167)
        pairs = [("OR", "QD", or_means, qd_means),
                 ("OR", "Def", or_means, def_means),
                 ("QD", "Def", qd_means, def_means)]

        alpha_corr = 0.05 / 3.0
        best_str = f"n.s. (p={p_kw:.3f})"

        if p_kw < 0.05:
            best_str = f"KW Signif (H={H:.2f}, p={p_kw:.3f})"
            if or_u > qd_u and or_u > def_u:
                best_str = f"OR > Baselines (p={p_kw:.3f})"

        for g1, g2, a, b_vals in pairs:
            U, p_raw = scipy_stats.mannwhitneyu(a, b_vals, alternative="two-sided")
            p_corr = min(p_raw * 3.0, 1.0)
            r_eff = rank_biserial(U, len(a), len(b_vals))
            pairwise_rows.append({
                "Cond": cond_id, "Pair": f"{g1} vs {g2}", "U": float(U),
                "p_raw": float(p_raw), "p_corr": float(p_corr), "r_effect": float(r_eff)
            })

        print(f"{cond_id:<6} {cond_params['n']:<4} {cond_params['obs_noise']:<6.1f} {cond_params['budget']:<5.2f} "
              f"{or_u:<10.1f} {qd_u:<10.1f} {def_u:<10.1f} {best_str:<25}")

        summary_rows.append({
            "Cond": cond_id, "n": cond_params["n"], "obs_noise": cond_params["obs_noise"],
            "budget": cond_params["budget"], "OR_mean": float(or_u), "QD_mean": float(qd_u),
            "Def_mean": float(def_u), "KW_H": float(H), "KW_p": float(p_kw)
        })

    # ── TABELA 8: PARÂMETROS APRENDIDOS PELO OR ───────────────────────────────
    print("\n" + "=" * 80)
    print("TABELA 8: Parâmetros OR Aprendidos por Condição (z_safe e beta)")
    print("=" * 80)
    print(f"{'Cond.':<6} {'n':<4} {'sigma':<6} {'b':<5} {'z_safe µ (σ)':<20} {'beta µ (σ)':<20}")
    print("-" * 80)

    table8_rows = []
    for cond_id, cond_params in conditions_dict.items():
        or_runs = all_results[cond_id]["OR"]
        sz_vals   = [r["or_sz_mean"]   for r in or_runs if "or_sz_mean" in r]
        beta_vals = [r["or_beta_mean"] for r in or_runs if "or_beta_mean" in r]

        sz_mu  = float(np.mean(sz_vals)) if sz_vals else 0.0
        sz_sd  = float(np.std(sz_vals)) if sz_vals else 0.0
        beta_mu = float(np.mean(beta_vals)) if beta_vals else 0.0
        beta_sd = float(np.std(beta_vals)) if beta_vals else 0.0

        sz_str   = f"{sz_mu:.3f} ({sz_sd:.3f})" if sz_vals else "N/A"
        beta_str = f"{beta_mu:.3f} ({beta_sd:.3f})" if beta_vals else "N/A"

        print(f"{cond_id:<6} {cond_params['n']:<4} {cond_params['obs_noise']:<6.1f} {cond_params['budget']:<5.2f} "
              f"{sz_str:<20} {beta_str:<20}")

        table8_rows.append({
            "Cond": cond_id, "n": cond_params["n"], "obs_noise": cond_params["obs_noise"],
            "budget": cond_params["budget"], "z_safe_mean": sz_mu, "z_safe_std": sz_sd,
            "beta_mean": beta_mu, "beta_std": beta_sd
        })

    # ── GERAR TODOS OS GRÁFICOS ───────────────────────────────────────────────
    plot_learning_curves(all_results, conditions_dict, timesteps, n_vars_target=args.dims[0], filename=f"learning_curves_n{args.dims[0]}.png")
    plot_learning_curves(all_results, conditions_dict, timesteps, n_vars_target=args.dims[1], filename=f"learning_curves_n{args.dims[1]}.png")
    plot_summary_barplots(all_results, conditions_dict)
    plot_learned_parameters(all_results, conditions_dict)

    # ── SALVAR TABELAS CSV E RESULTADOS ESTRUTURADOS ───────────────────────────
    with open("results/mrp_table6_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with open("results/mrp_table8_learned_params.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table8_rows[0].keys())
        writer.writeheader()
        writer.writerows(table8_rows)

    with open("results/mrp_pairwise_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_rows[0].keys())
        writer.writeheader()
        writer.writerows(pairwise_rows)

    with open("results/mrp_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "learned_params": table8_rows, "pairwise": pairwise_rows}, f, indent=2)

    print("\n -> Tabelas CSV e JSON salvos com sucesso na pasta 'results/'!")
    print(" -> Gráficos exportados com sucesso na pasta 'figures/'!")
    print("=" * 80)


if __name__ == "__main__":
    main()
