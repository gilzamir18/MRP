"""
run_ablation_or_fixed_experiment.py
====================================
Estudo de Ablação do Multi-Resource Problem (MRP):
Compara o Operational Regimes Adaptativo (OR) com o Operational Regimes Fixo (OR-Fixed),
além dos baselines contínuos QD e Standard Homeostatic (Default/DR).

Objetivo Científico (Resposta ao Revisor):
  Desacoplar os dois mecanismos do método:
  1. Estrutura Heurística em 3 Regimes (Maintenance, Recovery, Non-Recovery) [OR-Fixed].
  2. Auto-Sintonia Conjunta dos Hiperparâmetros (z_safe e beta via Policy Gradient) [OR].

Delineamento Experimental com Sementes Pareadas:
  - Todas as 4 variantes (OR, OR_fixed, QD, default) são treinadas com as MESMAS sementes.
  - Todas as 4 variantes são avaliadas na MESMA sequência exata de sementes de avaliação pareadas.
  - Todas as sementes usadas por rodada e episódio são registradas e salvas em CSV/JSON.
"""

import argparse
import csv
import json
import os
import random
import sys

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

from env import HomeostaticEnv

# Configurações visuais
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_palette("muted")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

VARIANTS = ["OR", "OR_fixed", "QD", "default"]
COLORS = {
    "OR": "#e05c5c",         # Vermelho (Adaptativo)
    "OR_fixed": "#e69138",   # Laranja/Âmbar (Ablação Fixo)
    "QD": "#4c8ef5",         # Azul (Differential Quadratic)
    "default": "#4cba6b",    # Verde (Standard Homeostatic / DR)
}
CURVE_POINTS = 100


def build_conditions(dims=(50, 100)):
    """Constrói o dicionário de condições fatoriais C1-C8."""
    n1, n2 = dims[0], dims[1]
    return {
        "C1": {"n": n1, "obs_noise": 0.0, "budget": 1.00, "name": f"n={n1}, no noise, b=1.0"},
        "C2": {"n": n1, "obs_noise": 0.2, "budget": 1.00, "name": f"n={n1}, noise=0.2, b=1.0"},
        "C3": {"n": n1, "obs_noise": 0.0, "budget": 0.20, "name": f"n={n1}, no noise, b=0.2"},
        "C4": {"n": n1, "obs_noise": 0.2, "budget": 0.20, "name": f"n={n1}, noise=0.2, b=0.2"},
        "C5": {"n": n2, "obs_noise": 0.0, "budget": 1.00, "name": f"n={n2}, no noise, b=1.0"},
        "C6": {"n": n2, "obs_noise": 0.2, "budget": 1.00, "name": f"n={n2}, noise=0.2, b=1.0"},
        "C7": {"n": n2, "obs_noise": 0.0, "budget": 0.20, "name": f"n={n2}, no noise, b=0.2"},
        "C8": {"n": n2, "obs_noise": 0.2, "budget": 0.20, "name": f"n={n2}, noise=0.2, b=0.2"},
    }


# ── WRAPPER DE PARÂMETROS APRENDIDOS PELO OR (PPO-CLIP) ───────────────────────
class ORParamWrapper(gym.Wrapper):
    """Wrapper que estende o espaço de ações com os parâmetros aprendidos z_safe e beta."""

    def __init__(self, env: HomeostaticEnv):
        super().__init__(env)
        ctrl_dim = env.action_space.shape[0]
        low = np.concatenate([env.action_space.low, [-5.0, -5.0]], dtype=np.float32)
        high = np.concatenate([env.action_space.high, [5.0, 5.0]], dtype=np.float32)
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
        sz = self.decode_sz(action[self._ctrl_dim])
        beta = self.decode_beta(action[self._ctrl_dim + 1])
        return self.env.step(ctrl, learned_params={"safety_zone": sz, "penalty_ratio": beta})


# ── REDE NEURAL ACTOR-CRITIC PARA BASELINES (OR_fixed, QD, default) ───────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
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
        mean = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def _compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    advantages, gae = [], 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


class _RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.__init__()

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

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
            ratio = torch.exp(new_lp - old_lp_t[idx])
            pg_loss = torch.max(-mb_adv * ratio,
                                -mb_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()

            vf_unclipped = (new_val - ret_t[idx]) ** 2
            vf_clipped = (old_val_t[idx] + torch.clamp(new_val - old_val_t[idx],
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


def interpolate_curve(ep_curve, total_timesteps, n_points=CURVE_POINTS, smooth_window=5):
    if not ep_curve:
        xs = np.linspace(0, total_timesteps, n_points)
        return xs, np.zeros(n_points)

    ts = np.array([t for t, _ in ep_curve], dtype=float)
    ys = np.array([l for _, l in ep_curve], dtype=float)

    if len(ys) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        ys_sm = np.convolve(ys, kernel, mode="same")
        for i in range(smooth_window // 2):
            ys_sm[i] = ys[:i + 1].mean()
            ys_sm[-(i + 1)] = ys[-(i + 1):].mean()
    else:
        ys_sm = ys.copy()

    xs_grid = np.linspace(ts[0], ts[-1], n_points)
    ys_grid = np.interp(xs_grid, ts, ys_sm)

    xs_full = np.linspace(0, total_timesteps, n_points)
    ys_full = np.zeros(n_points)
    idx_start = np.searchsorted(xs_full, xs_grid[0])
    idx_end = np.searchsorted(xs_full, xs_grid[-1])
    if idx_end >= n_points:
        idx_end = n_points - 1
    n_fill = idx_end - idx_start + 1
    ys_full[idx_start:idx_end + 1] = np.interp(
        np.linspace(xs_grid[0], xs_grid[-1], n_fill), xs_grid, ys_grid
    )
    ys_full[:idx_start] = ys_full[idx_start]
    ys_full[idx_end + 1:] = ys_full[idx_end]

    return xs_full, ys_full


# ── EXECUÇÃO DO EXPERIMENTO DE ABLAÇÃO COM SEMENTES PAREADAS ──────────────────
def run_single_experiment(
    variant: str, train_seed: int, cond_params: dict,
    total_timesteps: int, eval_seeds: list[int],
    fixed_z_safe: float, fixed_beta: float,
    learning_rate: float, n_steps: int, batch_size: int, n_epochs: int,
    hidden_dim: int, device: str
) -> dict:
    """
    Executa treino e avaliação de uma variante.
    Variantes suportadas:
      - "OR": Operational Regimes com aprendizado conjunto de z_safe e beta.
      - "OR_fixed": Operational Regimes com z_safe e beta fixos (sem aprendizado de hiperparâmetros).
      - "QD": Differential Quadratic.
      - "default": Standard Homeostatic (DR).
    """
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    random.seed(train_seed)

    num_vars = cond_params["n"]
    learn_or = (variant == "OR")
    reward_scale = 1.0 / num_vars

    # Para OR_fixed, usamos a recompensa OR mas com parâmetros fixados e use_learned_params=False
    rewarding_mode = "OR" if variant in ("OR", "OR_fixed") else variant

    env_kwargs = dict(
        num_vars=num_vars,
        rewarding=rewarding_mode,
        reward_scale=reward_scale,
        max_steps=1000,
        safety_zone=0.5 if variant != "OR_fixed" else fixed_z_safe,
        penalty_ratio=3.0 if variant != "OR_fixed" else fixed_beta,
        obs_noise_std=cond_params["obs_noise"],
        decay_std=0.001,
        decay_ratio_mu=0.001,
        correction_budget=cond_params["budget"],
        use_learned_params=learn_or,
    )

    ep_curve = []

    if learn_or:
        base_env = HomeostaticEnv(**env_kwargs)
        base_env.reset(seed=train_seed)
        train_env = ORParamWrapper(base_env)
        cb = EpisodeCurveCallback(total_timesteps)
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            policy_kwargs={"net_arch": [hidden_dim, hidden_dim]},
            seed=train_seed,
            device=device,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps, callback=cb)
        ep_curve = cb.ep_curve
        train_env.close()

    else:
        # Para OR_fixed, QD e default: usa rede ActorCritic padrão sem wrapper
        train_env = HomeostaticEnv(**env_kwargs)
        train_env.reset(seed=train_seed)
        obs_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        ac = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        optimizer = optim.Adam(ac.parameters(), lr=learning_rate, eps=1e-5)

        ep_curve = _train_ppo_custom(
            train_env, ac, optimizer,
            total_timesteps=total_timesteps,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            device=device,
        )
        train_env.close()
        ac.eval()

    # ── AVALIAÇÃO COM AS SEMENTES PAREADAS ────────────────────────────────────
    base_eval = HomeostaticEnv(**env_kwargs)
    eval_env = ORParamWrapper(base_eval) if learn_or else base_eval

    ep_lengths = []
    sz_samples = []
    beta_samples = []

    for seed_eval in eval_seeds:
        obs, _ = eval_env.reset(seed=seed_eval)
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
        "variant": variant,
        "train_seed": train_seed,
        "eval_seeds": eval_seeds,
        "ep_curve": ep_curve,
        "ep_lengths": ep_lengths,
        "mean_ep_len": float(np.mean(ep_lengths)),
        "std_ep_len": float(np.std(ep_lengths)),
        "median_ep_len": float(np.median(ep_lengths)),
    }

    if learn_or and sz_samples:
        result["or_sz_mean"] = float(np.mean(sz_samples))
        result["or_sz_std"] = float(np.std(sz_samples))
        result["or_beta_mean"] = float(np.mean(beta_samples))
        result["or_beta_std"] = float(np.std(beta_samples))

    return result


# ── GRÁFICOS DO ESTUDO DE ABLAÇÃO ─────────────────────────────────────────────
def plot_ablation_learning_curves(all_results_by_cond, conditions_dict, total_timesteps, n_vars_target, filename):
    os.makedirs("figures", exist_ok=True)
    cond_keys = [k for k, v in conditions_dict.items() if v["n"] == n_vars_target]
    if not cond_keys:
        return

    n_conds = len(cond_keys)
    n_cols = min(2, n_conds)
    n_rows = (n_conds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    xs_common = np.linspace(0, total_timesteps, CURVE_POINTS)

    for idx, cond in enumerate(cond_keys):
        ax = axes_flat[idx]
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
            std_curve = curves_arr.std(axis=0)

            color = COLORS[variant]
            label = "OR (Adaptive)" if variant == "OR" else "OR (Fixed)" if variant == "OR_fixed" else variant
            ls = "-" if variant != "OR_fixed" else "--"
            ax.plot(xs_common, mean_curve, label=label, color=color, linewidth=2, linestyle=ls)
            ax.fill_between(xs_common, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.15, color=color)

        info = conditions_dict[cond]
        ax.set_title(f"{cond}: n={info['n']}, noise={info['obs_noise']}, b={info['budget']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Training timesteps", fontsize=10)
        ax.set_ylabel("Mean Episode Length", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    for extra_idx in range(n_conds, len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    plt.suptitle(f"Ablation MRP Learning Curves (n = {n_vars_target} variables)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join("figures", filename)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" -> Gráfico salvo: {output_path}")


def plot_ablation_summary_barplots(all_results_by_cond, conditions_dict, output_filename="ablation_performance_comparison_bar.png"):
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    cond_labels = list(conditions_dict.keys())
    x = np.arange(len(cond_labels))
    width = 0.20

    means_by_var = {v: [] for v in VARIANTS}
    stds_by_var = {v: [] for v in VARIANTS}

    for cond in cond_labels:
        for v in VARIANTS:
            eps = [ep for r in all_results_by_cond[cond][v] for ep in r["ep_lengths"]]
            means_by_var[v].append(np.mean(eps))
            stds_by_var[v].append(np.std(eps))

    labels_map = {
        "OR": "OR (Adaptive)",
        "OR_fixed": "OR (Fixed Params)",
        "QD": "Differential Quadratic (QD)",
        "default": "Standard Homeostatic (DR)"
    }

    for i, v in enumerate(VARIANTS):
        offset = (i - 1.5) * width
        ax.bar(x + offset, means_by_var[v], width, yerr=stds_by_var[v],
               label=labels_map[v], color=COLORS[v], capsize=3, alpha=0.9)

    ax.set_ylabel('Mean Episode Length (Paired Evaluations)', fontsize=12, fontweight="bold")
    ax.set_xlabel('MRP Experimental Condition (C1 - C8)', fontsize=12, fontweight="bold")
    ax.set_title('MRP Ablation Study: Adaptive OR vs Fixed OR vs Baselines (Paired Seeds)', fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join("figures", output_filename)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" -> Gráfico comparativo de ablação salvo: {output_path}")


def plot_ablation_paired_differences(all_results_by_cond, conditions_dict, output_filename="ablation_paired_differences.png"):
    """Gera gráfico com as diferenças pareadas chave da ablação."""
    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    cond_labels = list(conditions_dict.keys())
    x = np.arange(len(cond_labels))

    d1_means, d1_stds = [], []  # OR vs OR_fixed (Ganho da adaptabilidade)
    d2_means, d2_stds = [], []  # OR_fixed vs QD (Ganho da heurística dos regimes)
    d3_means, d3_stds = [], []  # OR_fixed vs Default (Ganho da heurística dos regimes)

    for cond in cond_labels:
        or_eps = np.array([ep for r in all_results_by_cond[cond]["OR"] for ep in r["ep_lengths"]])
        or_f_eps = np.array([ep for r in all_results_by_cond[cond]["OR_fixed"] for ep in r["ep_lengths"]])
        qd_eps = np.array([ep for r in all_results_by_cond[cond]["QD"] for ep in r["ep_lengths"]])
        def_eps = np.array([ep for r in all_results_by_cond[cond]["default"] for ep in r["ep_lengths"]])

        d1 = or_eps - or_f_eps
        d2 = or_f_eps - qd_eps
        d3 = or_f_eps - def_eps

        d1_means.append(np.mean(d1)); d1_stds.append(np.std(d1))
        d2_means.append(np.mean(d2)); d2_stds.append(np.std(d2))
        d3_means.append(np.mean(d3)); d3_stds.append(np.std(d3))

    ax1.bar(x, d1_means, yerr=d1_stds, color='#e05c5c', alpha=0.85, capsize=3)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Adaptation Value:\nOR (Adaptive) - OR (Fixed)', fontsize=11, fontweight="bold")
    ax1.set_ylabel('Mean Paired Difference (steps)', fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(cond_labels)
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x, d2_means, yerr=d2_stds, color='#e69138', alpha=0.85, capsize=3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Regime Structure Value:\nOR (Fixed) - QD', fontsize=11, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(cond_labels)
    ax2.grid(True, axis='y', alpha=0.3)

    ax3.bar(x, d3_means, yerr=d3_stds, color='#4cba6b', alpha=0.85, capsize=3)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('Regime Structure Value:\nOR (Fixed) - Default', fontsize=11, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(cond_labels)
    ax3.grid(True, axis='y', alpha=0.3)

    plt.suptitle("MRP Ablation Paired Decompositions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join("figures", output_filename)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" -> Gráfico de diferenças pareadas de ablação salvo: {output_path}")


def plot_ablation_learned_parameters(all_results_by_cond, conditions_dict, fixed_z_safe, fixed_beta,
                                     output_filename="ablation_learned_parameters.png"):
    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    conds = list(conditions_dict.keys())
    sz_means, sz_stds = [], []
    beta_means, beta_stds = [], []

    for cond in conds:
        runs = all_results_by_cond[cond]["OR"]
        sz_vals = [r["or_sz_mean"] for r in runs if "or_sz_mean" in r]
        beta_vals = [r["or_beta_mean"] for r in runs if "or_beta_mean" in r]

        sz_means.append(np.mean(sz_vals) if sz_vals else 0)
        sz_stds.append(np.std(sz_vals) if sz_vals else 0)
        beta_means.append(np.mean(beta_vals) if beta_vals else 0)
        beta_stds.append(np.std(beta_vals) if beta_vals else 0)

    x = np.arange(len(conds))

    ax1.bar(x, sz_means, yerr=sz_stds, color='#8884d8', capsize=4, alpha=0.85, label='Learned z_safe (OR)')
    ax1.axhline(fixed_z_safe, color='#e69138', linestyle='--', linewidth=2, label=f'Fixed z_safe = {fixed_z_safe:.2f}')
    ax1.set_title('Learned Safety Zone (z_safe) vs Fixed Baseline', fontsize=12, fontweight="bold")
    ax1.set_ylabel('z_safe (half-width)', fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(conds)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x, beta_means, yerr=beta_stds, color='#82ca9d', capsize=4, alpha=0.85, label='Learned beta (OR)')
    ax2.axhline(fixed_beta, color='#e69138', linestyle='--', linewidth=2, label=f'Fixed beta = {fixed_beta:.2f}')
    ax2.set_title('Learned Penalty Ratio (beta) vs Fixed Baseline', fontsize=12, fontweight="bold")
    ax2.set_ylabel('beta (penalty ratio)', fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(conds)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle("Jointly Learned Parameters (OR) vs Fixed Ablation Reference", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join("figures", output_filename)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" -> Gráfico de parâmetros de ablação salvo: {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Estudo de Ablação do MRP: OR Adaptativo vs OR Fixo (Sementes Pareadas)")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Número de episódios de avaliação pareados (default: 8)")
    parser.add_argument("--eval-seed-start", type=int, default=1000, help="Semente inicial para a sequência pareada de avaliação")
    parser.add_argument("--seeds-per-cond", type=int, default=3, help="Número de sementes de treino pareadas por condição (default: 3)")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps de treino por variante (default: 50000)")
    parser.add_argument("--dims", type=int, nargs=2, default=[50, 100], help="Dimensões n1 e n2 (default: 50 100)")
    parser.add_argument("--conditions", type=str, default=None, help="Lista de condições separadas por vírgula (ex: C1,C2 ou default: todas C1-C8)")
    parser.add_argument("--fixed-mode", type=str, default="canonical",
                        choices=["canonical", "empirical_mean", "asymmetric", "custom"],
                        help="Modo de definição dos parâmetros fixos: 'canonical' (z=0.5, b=1.0), 'empirical_mean' (z=0.52, b=0.67), 'asymmetric' (z=0.5, b=3.0), 'custom'")
    parser.add_argument("--fixed-z-safe", type=float, default=0.5, help="Valor fixo de z_safe quando fixed-mode=custom (default: 0.5)")
    parser.add_argument("--fixed-beta", type=float, default=1.0, help="Valor fixo de beta quando fixed-mode=custom (default: 1.0)")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo PyTorch ('cpu' ou 'cuda', default: cpu)")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout steps")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    args = parser.parse_args()

    # Define os parâmetros fixos conforme o modo
    if args.fixed_mode == "canonical":
        fixed_sz = 0.50
        fixed_beta = 1.00
    elif args.fixed_mode == "empirical_mean":
        fixed_sz = 0.52
        fixed_beta = 0.67
    elif args.fixed_mode == "asymmetric":
        fixed_sz = 0.50
        fixed_beta = 3.00
    else:  # custom
        fixed_sz = args.fixed_z_safe
        fixed_beta = args.fixed_beta

    all_conds = build_conditions(args.dims)
    if args.conditions:
        selected_keys = [c.strip().upper() for c in args.conditions.split(",") if c.strip().upper() in all_conds]
        conditions_dict = {k: all_conds[k] for k in selected_keys}
    else:
        conditions_dict = all_conds

    train_seeds = list(range(42, 42 + args.seeds_per_cond))
    eval_seeds = [args.eval_seed_start + i for i in range(args.eval_episodes)]

    print("=" * 80)
    print("  ESTUDO DE ABLAÇÃO: OPERATIONAL REGIMES ADAPTATIVO VS FIXO (MRP)")
    print("  DELINEAMENTO DE SEMENTES PAREADAS E REGISTRO COMPLETO DE SEEDS")
    print(f"  Variantes Avaliadas: {VARIANTS}")
    print(f"  Configuração OR-Fixed ({args.fixed_mode}): z_safe = {fixed_sz:.2f}, beta = {fixed_beta:.2f}")
    print(f"  Dimensões Testadas: {args.dims[0]} e {args.dims[1]} variáveis")
    print(f"  Seeds de Treino Pareadas: {train_seeds}")
    print(f"  {len(eval_seeds)} Episódios de Avaliação Pareados (Seeds: {eval_seeds})")
    print(f"  Timesteps de Treino: {args.timesteps} | Dispositivo: {args.device}")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    all_results = {}

    for cond_id, cond_params in conditions_dict.items():
        print(f"\n>>> Executando Condição {cond_id}: {cond_params['name']} ...")
        all_results[cond_id] = {v: [] for v in VARIANTS}

        for variant in VARIANTS:
            for seed in train_seeds:
                var_label = "OR (Adapt)" if variant == "OR" else "OR (Fixed)" if variant == "OR_fixed" else variant
                print(f"  [{cond_id}] {var_label:<12} seed={seed} (eval pareados) ... ", end="", flush=True)
                res = run_single_experiment(
                    variant=variant,
                    train_seed=seed,
                    cond_params=cond_params,
                    total_timesteps=args.timesteps,
                    eval_seeds=eval_seeds,
                    fixed_z_safe=fixed_sz,
                    fixed_beta=fixed_beta,
                    learning_rate=3e-4,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    hidden_dim=64,
                    device=args.device,
                )
                all_results[cond_id][variant].append(res)
                print(f"-> mean={res['mean_ep_len']:.1f} (std={res['std_ep_len']:.1f})", end="")
                if "or_sz_mean" in res:
                    print(f" | OR sz={res['or_sz_mean']:.3f} beta={res['or_beta_mean']:.3f}", end="")
                print()

    # ── TABELA DE RESULTADOS ESTATÍSTICOS DA ABLAÇÃO ──────────────────────────
    print("\n" + "=" * 105)
    print("TABELA ABLAÇÃO: Duração de Episódio Pareada e Testes de Hipótese (4 Variantes)")
    print("=" * 105)
    print(f"{'Cond.':<6} {'n':<4} {'noise':<6} {'b':<5} {'OR_Adapt (µ±σ)':<16} {'OR_Fixed (µ±σ)':<16} {'QD (µ±σ)':<14} {'Def (µ±σ)':<14} {'Wilcoxon (OR vs OR_fix)':<20}")
    print("-" * 105)

    summary_rows = []
    paired_rows = []

    for cond_id, cond_params in conditions_dict.items():
        or_eps = np.array([ep for r in all_results[cond_id]["OR"] for ep in r["ep_lengths"]])
        or_f_eps = np.array([ep for r in all_results[cond_id]["OR_fixed"] for ep in r["ep_lengths"]])
        qd_eps = np.array([ep for r in all_results[cond_id]["QD"] for ep in r["ep_lengths"]])
        def_eps = np.array([ep for r in all_results[cond_id]["default"] for ep in r["ep_lengths"]])

        or_mu, or_sd = float(np.mean(or_eps)), float(np.std(or_eps))
        orf_mu, orf_sd = float(np.mean(or_f_eps)), float(np.std(or_f_eps))
        qd_mu, qd_sd = float(np.mean(qd_eps)), float(np.std(qd_eps))
        def_mu, def_sd = float(np.mean(def_eps)), float(np.std(def_eps))

        # Teste Omnibus de Friedman com 4 tratamentos pareados
        try:
            stat_friedman, p_friedman = scipy_stats.friedmanchisquare(or_eps, or_f_eps, qd_eps, def_eps)
        except Exception:
            stat_friedman, p_friedman = float('nan'), float('nan')

        # Função auxiliar de Wilcoxon pareado
        def calc_wilcoxon(a, b):
            diff = a - b
            if np.all(diff == 0):
                return 0.0, 1.0, 0.0, 0.0
            stat, p_val = scipy_stats.wilcoxon(a, b, alternative="two-sided")
            return float(stat), float(p_val), float(np.mean(diff)), float(np.std(diff))

        pairs_to_test = [
            ("OR vs OR_fixed", or_eps, or_f_eps),
            ("OR vs QD", or_eps, qd_eps),
            ("OR vs Default", or_eps, def_eps),
            ("OR_fixed vs QD", or_f_eps, qd_eps),
            ("OR_fixed vs Default", or_f_eps, def_eps),
            ("QD vs Default", qd_eps, def_eps),
        ]

        n_pairs = len(pairs_to_test)
        p_or_v_orf = 1.0

        for pair_name, v1, v2 in pairs_to_test:
            w_stat, p_val, d_mean, d_sd = calc_wilcoxon(v1, v2)
            p_bonf = min(p_val * n_pairs, 1.0)
            paired_rows.append({
                "Cond": cond_id, "Pair": pair_name, "N_paired": len(v1),
                "Mean_Diff": d_mean, "Std_Diff": d_sd,
                "Wilcoxon_W": w_stat, "p_raw": p_val, "p_bonferroni": p_bonf,
            })
            if pair_name == "OR vs OR_fixed":
                p_or_v_orf = p_val
                d_or_orf_mean = d_mean

        test_str = f"p={p_or_v_orf:.3f} (Δ={d_or_orf_mean:+.1f})"

        print(f"{cond_id:<6} {cond_params['n']:<4} {cond_params['obs_noise']:<6.1f} {cond_params['budget']:<5.2f} "
              f"{or_mu:>5.1f}±{or_sd:<9.1f} {orf_mu:>5.1f}±{orf_sd:<9.1f} {qd_mu:>5.1f}±{qd_sd:<7.1f} {def_mu:>5.1f}±{def_sd:<7.1f} "
              f"{test_str:<20}")

        summary_rows.append({
            "Cond": cond_id, "n": cond_params["n"], "obs_noise": cond_params["obs_noise"],
            "budget": cond_params["budget"],
            "OR_Adapt_mean": or_mu, "OR_Adapt_std": or_sd,
            "OR_Fixed_mean": orf_mu, "OR_Fixed_std": orf_sd,
            "QD_mean": qd_mu, "QD_std": qd_sd,
            "Def_mean": def_mu, "Def_std": def_sd,
            "Friedman_stat": float(stat_friedman), "Friedman_p": float(p_friedman),
            "Wilcoxon_OR_v_OR_fixed_p": p_or_v_orf,
            "Diff_OR_vs_OR_fixed_mean": d_or_orf_mean,
        })

    # ── TABELA DE PARÂMETROS APRENDIDOS PELO OR ───────────────────────────────
    print("\n" + "=" * 80)
    print("TABELA PARÂMETROS OR APRENDIDOS vs FIXADOS")
    print("=" * 80)
    print(f"{'Cond.':<6} {'n':<4} {'noise':<6} {'b':<5} {'Learned z_safe (σ)':<20} {'Fixed z_safe':<14} {'Learned beta (σ)':<20} {'Fixed beta':<12}")
    print("-" * 80)

    table8_rows = []
    for cond_id, cond_params in conditions_dict.items():
        or_runs = all_results[cond_id]["OR"]
        sz_vals = [r["or_sz_mean"] for r in or_runs if "or_sz_mean" in r]
        beta_vals = [r["or_beta_mean"] for r in or_runs if "or_beta_mean" in r]

        sz_mu = float(np.mean(sz_vals)) if sz_vals else 0.0
        sz_sd = float(np.std(sz_vals)) if sz_vals else 0.0
        beta_mu = float(np.mean(beta_vals)) if beta_vals else 0.0
        beta_sd = float(np.std(beta_vals)) if beta_vals else 0.0

        sz_str = f"{sz_mu:.3f} ({sz_sd:.3f})" if sz_vals else "N/A"
        beta_str = f"{beta_mu:.3f} ({beta_sd:.3f})" if beta_vals else "N/A"

        print(f"{cond_id:<6} {cond_params['n']:<4} {cond_params['obs_noise']:<6.1f} {cond_params['budget']:<5.2f} "
              f"{sz_str:<20} {fixed_sz:<14.2f} {beta_str:<20} {fixed_beta:<12.2f}")

        table8_rows.append({
            "Cond": cond_id, "n": cond_params["n"], "obs_noise": cond_params["obs_noise"],
            "budget": cond_params["budget"],
            "z_safe_learned_mean": sz_mu, "z_safe_learned_std": sz_sd, "z_safe_fixed": fixed_sz,
            "beta_learned_mean": beta_mu, "beta_learned_std": beta_sd, "beta_fixed": fixed_beta,
        })

    # ── LOG DETALHADO DE SEMENTES PAREADAS POR RODADA / EPISÓDIO ────────────
    seed_log_rows = []
    for cond_id in conditions_dict.keys():
        for s_idx, t_seed in enumerate(train_seeds):
            or_run = all_results[cond_id]["OR"][s_idx]
            orf_run = all_results[cond_id]["OR_fixed"][s_idx]
            qd_run = all_results[cond_id]["QD"][s_idx]
            def_run = all_results[cond_id]["default"][s_idx]
            for ep_i, e_seed in enumerate(eval_seeds):
                or_len = or_run["ep_lengths"][ep_i]
                orf_len = orf_run["ep_lengths"][ep_i]
                qd_len = qd_run["ep_lengths"][ep_i]
                def_len = def_run["ep_lengths"][ep_i]
                seed_log_rows.append({
                    "Cond": cond_id,
                    "Train_Seed": t_seed,
                    "Ep_Index": ep_i + 1,
                    "Eval_Seed": e_seed,
                    "OR_Adapt_ep_len": or_len,
                    "OR_Fixed_ep_len": orf_len,
                    "QD_ep_len": qd_len,
                    "Def_ep_len": def_len,
                    "Diff_OR_vs_OR_fixed": or_len - orf_len,
                    "Diff_OR_vs_QD": or_len - qd_len,
                    "Diff_OR_fixed_vs_Def": orf_len - def_len,
                })

    # ── GERAÇÃO DOS GRÁFICOS ──────────────────────────────────────────────────
    plot_ablation_learning_curves(all_results, conditions_dict, args.timesteps, n_vars_target=args.dims[0],
                                  filename=f"ablation_learning_curves_n{args.dims[0]}.png")
    plot_ablation_learning_curves(all_results, conditions_dict, args.timesteps, n_vars_target=args.dims[1],
                                  filename=f"ablation_learning_curves_n{args.dims[1]}.png")
    plot_ablation_summary_barplots(all_results, conditions_dict, output_filename="ablation_performance_comparison_bar.png")
    plot_ablation_paired_differences(all_results, conditions_dict, output_filename="ablation_paired_differences.png")
    plot_ablation_learned_parameters(all_results, conditions_dict, fixed_sz, fixed_beta,
                                     output_filename="ablation_learned_parameters.png")

    # ── EXPORTAÇÃO DOS ARQUIVOS CSV E JSON ────────────────────────────────────
    with open("results/ablation_seed_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seed_log_rows[0].keys())
        writer.writeheader()
        writer.writerows(seed_log_rows)

    with open("results/ablation_table6_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with open("results/ablation_table8_learned_params.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table8_rows[0].keys())
        writer.writeheader()
        writer.writerows(table8_rows)

    with open("results/ablation_paired_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paired_rows[0].keys())
        writer.writeheader()
        writer.writerows(paired_rows)

    with open("results/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "fixed_mode": args.fixed_mode,
                "fixed_z_safe": fixed_sz,
                "fixed_beta": fixed_beta,
                "eval_episodes": args.eval_episodes,
                "eval_seeds": eval_seeds,
                "train_seeds": train_seeds,
                "timesteps": args.timesteps,
                "dims": args.dims,
                "device": args.device
            },
            "summary": summary_rows,
            "learned_params": table8_rows,
            "paired_stats": paired_rows,
            "seed_log": seed_log_rows
        }, f, indent=2)

    print("\n" + "=" * 85)
    print(" -> Tabelas de Ablação salvas em 'results/':")
    print("    • results/ablation_seed_log.csv (Log linha-a-linha de sementes pareadas)")
    print("    • results/ablation_table6_summary.csv")
    print("    • results/ablation_table8_learned_params.csv")
    print("    • results/ablation_paired_stats.csv")
    print("    • results/ablation_results.json")
    print(" -> Gráficos de Ablação salvos em 'figures/':")
    print(f"    • figures/ablation_learning_curves_n{args.dims[0]}.png")
    print(f"    • figures/ablation_learning_curves_n{args.dims[1]}.png")
    print("    • figures/ablation_performance_comparison_bar.png")
    print("    • figures/ablation_paired_differences.png")
    print("    • figures/ablation_learned_parameters.png")
    print("=" * 85)


if __name__ == "__main__":
    main()
