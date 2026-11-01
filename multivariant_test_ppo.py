"""
multivariant_test_ppo_v2.py
===========================
Igual ao multivariant_test_ppo.py, mas adiciona o plot da curva de aprendizado:
  • Eixo X: passos de treino (timesteps)
  • Eixo Y: tamanho médio de episódio (média móvel sobre janela deslizante)
  • Três linhas: OR, QD, default
  • Banda sombreada: ± 1 desvio padrão entre seeds

Uso:
  python multivariant_test_ppo_v2.py
  python multivariant_test_ppo_v2.py --train-seeds 5 --eval-episodes 30
  python multivariant_test_ppo_v2.py --test-mode
  python multivariant_test_ppo_v2.py --plot-only --results-file results/curves.npy
"""

import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from env import HomeostaticEnv

# ── constantes ────────────────────────────────────────────────────────────────
NUM_VARS         = 50
DEFAULT_TRAIN_K  = 2
DEFAULT_EVAL_L   = 10
EVAL_SEED_START  = 1_000
DEFAULT_TS       = 300_000
SAFETY_ZONE      = 0.25
MAX_STEPS        = 1_000
ALPHA            = 0.05

VARIANTS         = ["OR", "QD", "default"]
REWARD_SCALE     = {"QD": 1.0, "OR": 1.0, "default": 1.0}
COLORS           = {"OR": "#e05c5c", "QD": "#4c8ef5", "default": "#4cba6b"}

# Número de pontos no eixo X do gráfico após interpolação
CURVE_POINTS     = 200


# ═════════════════════════════════════════════════════════════════════════════
# ORParamWrapper  (SB3-path, para OR)
# ═════════════════════════════════════════════════════════════════════════════

class ORParamWrapper(gym.Wrapper):
    """Estende action_space com sz_raw e beta_raw aprendidos pelo SB3-PPO."""

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


# ═════════════════════════════════════════════════════════════════════════════
# PPO customizado  (ppo2.py — para QD e default)
# ═════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """Actor-Critic com tronco MLP compartilhado."""

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
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
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

    def __len__(self): return len(self.rewards)


def _ppo_update(
    ac, optimizer, obs_t, actions_t, old_lp_t, adv_t, ret_t, old_val_t,
    n_epochs, batch_size, clip_range, vf_coef, ent_coef, max_grad_norm,
):
    n = obs_t.shape[0]
    for _ in range(n_epochs):
        for start in range(0, n, batch_size):
            idx    = torch.randperm(n)[start: start + batch_size]
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


def _train_ppo_custom(
    env, ac, optimizer, total_timesteps, n_steps, n_epochs, batch_size,
    gamma, gae_lambda, clip_range, vf_coef, ent_coef, max_grad_norm, device,
) -> list[tuple[int, int]]:
    """Treina com PPO customizado. Retorna lista de (timestep_fim_ep, ep_length)."""
    buffer    = _RolloutBuffer()
    obs, _    = env.reset()
    done = truncated = False
    total_steps  = 0
    ep_len       = 0
    ep_curve: list[tuple[int, int]] = []   # (timestep, ep_length)

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
            ep_len      += 1
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
            last_obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(device)
            _, _, last_v = ac.get_action(last_obs_t)
            last_value   = last_v.item() if not (done or truncated) else 0.0

        advantages, returns = _compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            last_value, gamma, gae_lambda,
        )
        obs_t, act_t, lp_t, adv_t, ret_t = buffer.to_tensors(advantages, returns, device)
        old_val_t = torch.FloatTensor(buffer.values).to(device)

        _ppo_update(ac, optimizer, obs_t, act_t, lp_t, adv_t, ret_t, old_val_t,
                    n_epochs, batch_size, clip_range, vf_coef, ent_coef, max_grad_norm)

    pbar.close()
    return ep_curve


# ═════════════════════════════════════════════════════════════════════════════
# Callback SB3 para coletar curva de treino (OR)
# ═════════════════════════════════════════════════════════════════════════════

class EpisodeCurveCallback(BaseCallback):
    """Coleta (timestep_fim_ep, ep_length) durante .learn() do SB3."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self._total     = total_timesteps
        self._pbar: tqdm | None = None
        self.ep_curve: list[tuple[int, int]] = []
        self._ep_len  = 0

    def _on_training_start(self) -> None:
        self._pbar = tqdm(total=self._total, desc="  treino", leave=False, unit="step")

    def _on_step(self) -> bool:
        n = self.training_env.num_envs
        self._pbar.update(n)
        self._ep_len += n

        # locals()["dones"] é um array booleano de shape (n_envs,)
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self.ep_curve.append((self.num_timesteps, self._ep_len))
                self._ep_len = 0
        return True

    def _on_training_end(self) -> None:
        self._pbar.close()


# ═════════════════════════════════════════════════════════════════════════════
# Utilitário: interpolação em grade comum
# ═════════════════════════════════════════════════════════════════════════════

def interpolate_curve(
    ep_curve: list[tuple[int, int]],
    total_timesteps: int,
    n_points: int = CURVE_POINTS,
    smooth_window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recebe lista de (timestep, ep_length), aplica média móvel e interpola
    em grade uniforme de n_points pontos entre 0 e total_timesteps.
    Retorna (xs, ys).
    """
    if not ep_curve:
        xs = np.linspace(0, total_timesteps, n_points)
        return xs, np.zeros(n_points)

    ts  = np.array([t for t, _ in ep_curve], dtype=float)
    ys  = np.array([l for _, l in ep_curve], dtype=float)

    # média móvel simples sobre os episódios
    if len(ys) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        ys_sm  = np.convolve(ys, kernel, mode="same")
        # corrige bordas (janela incompleta)
        for i in range(smooth_window // 2):
            ys_sm[i]            = ys[:i+1].mean()
            ys_sm[-(i+1)]       = ys[-(i+1):].mean()
    else:
        ys_sm = ys.copy()

    xs_grid = np.linspace(ts[0], ts[-1], n_points)
    ys_grid = np.interp(xs_grid, ts, ys_sm)

    # Coloca no domínio completo de treino
    xs_full = np.linspace(0, total_timesteps, n_points)
    ys_full = np.zeros(n_points)
    # mapeia xs_grid → posições em xs_full
    idx_start = np.searchsorted(xs_full, xs_grid[0])
    idx_end   = np.searchsorted(xs_full, xs_grid[-1])
    if idx_end >= n_points:
        idx_end = n_points - 1
    n_fill = idx_end - idx_start + 1
    ys_full[idx_start:idx_end+1] = np.interp(
        np.linspace(xs_grid[0], xs_grid[-1], n_fill),
        xs_grid, ys_grid,
    )
    # preenche antes do primeiro e após o último episódio com valor da borda
    ys_full[:idx_start] = ys_full[idx_start]
    ys_full[idx_end+1:] = ys_full[idx_end]

    return xs_full, ys_full


# ═════════════════════════════════════════════════════════════════════════════
# Experimento unificado
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(
    variant: str,
    train_seed: int,
    num_vars: int,
    max_steps: int,
    total_timesteps: int,
    n_eval_episodes: int,
    eval_seed_start: int,
    safety_zone: float,
    penalty_ratio: float,
    obs_noise_std: float,
    correction_budget: float,
    decay_std: float,
    decay_ratio_mu: float,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    hidden_dim: int,
    device: str,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
) -> dict:
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    random.seed(train_seed)

    learn_or = (variant == "OR")

    env_kwargs = dict(
        num_vars           = num_vars,
        rewarding          = variant,
        reward_scale       = REWARD_SCALE[variant],
        max_steps          = max_steps,
        safety_zone        = safety_zone,
        penalty_ratio      = penalty_ratio,
        obs_noise_std      = obs_noise_std,
        decay_std          = decay_std,
        decay_ratio_mu     = decay_ratio_mu,
        correction_budget  = correction_budget,
        use_learned_params = learn_or,
    )

    ep_curve: list[tuple[int, int]] = []

    # ── Treino ────────────────────────────────────────────────────────────────
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
            gamma           = gamma,
            gae_lambda      = gae_lambda,
            clip_range      = clip_range,
            vf_coef         = vf_coef,
            ent_coef        = ent_coef,
            max_grad_norm   = max_grad_norm,
            device          = device,
        )
        train_env.close()
        ac.eval()

    # ── Avaliação ─────────────────────────────────────────────────────────────
    base_eval = HomeostaticEnv(**env_kwargs)
    eval_env  = ORParamWrapper(base_eval) if learn_or else base_eval

    ep_lengths:   list[int]   = []
    sz_samples:   list[float] = []
    beta_samples: list[float] = []

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

    result: dict = {
        "variant"       : variant,
        "train_seed"    : train_seed,
        "ep_curve"      : ep_curve,           # ← novo
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


# ═════════════════════════════════════════════════════════════════════════════
# Utilitários estatísticos
# ═════════════════════════════════════════════════════════════════════════════

def rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * u) / (n1 * n2)


def effect_label(r: float) -> str:
    ar = abs(r)
    if ar < 0.10: return "negligível"
    if ar < 0.30: return "pequeno"
    if ar < 0.50: return "médio"
    return "grande"


def statistical_report(variant_means: dict[str, list[float]], alpha: float = ALPHA) -> None:
    groups  = list(variant_means.keys())
    arrays  = [variant_means[g] for g in groups]
    pairs   = list(itertools.combinations(range(len(groups)), 2))
    n_pairs = len(pairs)
    alpha_b = alpha / n_pairs

    print(f"\n{'-'*55}")
    print("  Kruskal-Wallis  (H0: distribuicoes identicas entre grupos)")
    print(f"{'-'*55}")

    if any(len(a) < 2 for a in arrays):
        print("  AVISO: k < 2 para algum grupo — testes sem poder estatistico.")
        print("  Use --train-seeds >= 5 para resultados confiaveis.\n")
        return

    H, p_kw = scipy_stats.kruskal(*arrays)
    sig_kw  = p_kw < alpha
    print(f"  H = {H:.4f}   p = {p_kw:.4f}   {'*** SIGNIFICATIVO' if sig_kw else '(nao significativo)'}")
    print(f"  alpha = {alpha}   grupos: {groups}")

    if not sig_kw:
        print("\n  -> KW nao rejeita H0: sem evidencia de diferencia entre grupos.")
        print("    Testes par-a-par omitidos.\n")
        return

    print(f"\n{'-'*55}")
    print(f"  Mann-Whitney U  (par-a-par, Bonferroni alpha_corr = {alpha_b:.4f})")
    print(f"{'-'*55}")
    print(f"  {'Par':<18} {'U':>8} {'p_raw':>9} {'p_corr':>9} {'Sig':>4} {'r':>7} {'Efeito'}")
    print(f"  {'':-<18} {'':->8} {'':->9} {'':->9} {'':->4} {'':->7} {'':-<10}")

    for i, j in pairs:
        a, b   = arrays[i], arrays[j]
        label  = f"{groups[i]} vs {groups[j]}"
        U, p_r = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        p_c    = min(p_r * n_pairs, 1.0)
        r      = rank_biserial(U, len(a), len(b))
        sig    = "***" if p_c < alpha else "ns"
        print(f"  {label:<18} {U:>8.1f} {p_r:>9.4f} {p_c:>9.4f} {sig:>4} {r:>7.3f} {effect_label(r)}")

    print(f"\n  Interpretacao de r:  |r|<0.1 negligivel | 0.1 pequeno | 0.3 medio | 0.5 grande")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Plot das curvas de aprendizado
# ═════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(
    all_results: dict[str, list[dict]],
    total_timesteps: int,
    output_path: str = "figures/learning_curves.png",
    smooth_window: int = 10,
    n_points: int = CURVE_POINTS,
) -> None:
    """
    Plota média ± desvio padrão do tamanho de episódio ao longo do treino
    para cada variante.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    xs_common = np.linspace(0, total_timesteps, n_points)

    for variant in VARIANTS:
        runs = all_results[variant]
        if not runs:
            continue

        curves = []
        for res in runs:
            _, ys = interpolate_curve(
                res["ep_curve"], total_timesteps,
                n_points=n_points, smooth_window=smooth_window,
            )
            curves.append(ys)

        curves_arr = np.array(curves)           # shape (n_seeds, n_points)
        mean_curve = curves_arr.mean(axis=0)
        std_curve  = curves_arr.std(axis=0)

        color = COLORS[variant]
        ax.plot(xs_common, mean_curve, label=variant, color=color, linewidth=2)
        ax.fill_between(
            xs_common,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2, color=color,
        )

    ax.set_xlabel("Training steps (timesteps)", fontsize=12)
    ax.set_ylabel("Mean episode length", fontsize=12)
    ax.set_title(
        "Evolution of episode length during training\n"
        f"(mean ± deviation between seeds, smooth window={smooth_window})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n  Gráfico salvo em: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comparação multi-variante PPO híbrido com curva de aprendizado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-seeds",    type=int,   default=DEFAULT_TRAIN_K)
    parser.add_argument("--eval-episodes",  type=int,   default=DEFAULT_EVAL_L)
    parser.add_argument("--timesteps",      type=int,   default=DEFAULT_TS)
    parser.add_argument("--max-steps",      type=int,   default=MAX_STEPS)
    parser.add_argument("--noise",          type=float, default=0.0)
    parser.add_argument("--budget",         type=float, default=1.0)
    parser.add_argument("--n-steps",        type=int,   default=2048)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--n-epochs",       type=int,   default=10)
    parser.add_argument("--hidden-dim",     type=int,   default=64)
    parser.add_argument("--learning-rate",  type=float, default=3e-4)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--gae-lambda",     type=float, default=0.95)
    parser.add_argument("--clip-range",     type=float, default=0.2)
    parser.add_argument("--vf-coef",        type=float, default=0.5)
    parser.add_argument("--ent-coef",       type=float, default=0.0)
    parser.add_argument("--max-grad-norm",  type=float, default=0.5)
    parser.add_argument("--device",         type=str,   default="cpu")
    parser.add_argument("--alpha",          type=float, default=ALPHA)
    parser.add_argument("--smooth-window",  type=int,   default=10,
                        help="Janela da média móvel para suavizar a curva")
    parser.add_argument("--output",         type=str,   default="figures/learning_curves.png",
                        help="Caminho para salvar o gráfico")
    parser.add_argument("--test-mode",      action="store_true",
                        help="2 seeds, 20k steps, n_vars=4, 10 eval episodes")
    args = parser.parse_args()

    train_seeds   = list(range(42, 42 + args.train_seeds))
    eval_episodes = args.eval_episodes
    timesteps     = args.timesteps
    n_vars        = NUM_VARS

    if args.test_mode:
        train_seeds   = [42, 43]
        eval_episodes = 10
        timesteps     = 20_000
        n_vars        = 4

    total = len(VARIANTS) * len(train_seeds)

    print(f"\n{'='*55}")
    print(f"  OR (SB3)  vs  QD (custom)  vs  default (custom)  [PPO híbrido]")
    print(f"  train_seeds={train_seeds}  eval_episodes={eval_episodes}")
    print(f"  timesteps={timesteps}  n_vars={n_vars}")
    print(f"  eval_seeds={EVAL_SEED_START}…{EVAL_SEED_START + eval_episodes - 1}")
    print(f"{'='*55}\n")

    all_results: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    done_count = 0

    for variant in VARIANTS:
        for seed in train_seeds:
            done_count += 1
            print(f"[{done_count}/{total}]  {variant:<8}  seed={seed}  ...", flush=True)
            res = run_experiment(
                variant           = variant,
                train_seed        = seed,
                num_vars          = n_vars,
                max_steps         = args.max_steps,
                total_timesteps   = timesteps,
                n_eval_episodes   = eval_episodes,
                eval_seed_start   = EVAL_SEED_START,
                safety_zone       = SAFETY_ZONE,
                penalty_ratio     = 3.0,
                obs_noise_std     = args.noise,
                correction_budget = args.budget,
                decay_std         = 0.001,
                decay_ratio_mu    = 0.001,
                learning_rate     = args.learning_rate,
                n_steps           = args.n_steps,
                batch_size        = args.batch_size,
                n_epochs          = args.n_epochs,
                hidden_dim        = args.hidden_dim,
                device            = args.device,
                gamma             = args.gamma,
                gae_lambda        = args.gae_lambda,
                clip_range        = args.clip_range,
                vf_coef           = args.vf_coef,
                ent_coef          = args.ent_coef,
                max_grad_norm     = args.max_grad_norm,
            )
            all_results[variant].append(res)
            n_ep_train = len(res["ep_curve"])
            print(f"  -> mean={res['mean_ep_len']:.1f}  median={res['median_ep_len']:.1f}"
                  f"  ep_treino={n_ep_train}", end="")
            if "or_sz_mean" in res:
                print(
                    f"  sz={res['or_sz_mean']:.3f}+-{res['or_sz_std']:.3f}"
                    f"  beta={res['or_beta_mean']:.3f}+-{res['or_beta_std']:.3f}",
                    end="",
                )
            print()

    # ── Resumo tabular ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Resumo - episode_length  (cada valor = media sobre {eval_episodes} eps)")
    print(f"{'='*55}")
    print(f"  {'Variante':<10} {'mu(mean)':>9} {'std(mean)':>9} {'med(mean)':>10}"
          f"  {'mu(median)':>10} {'med(median)':>11}")
    print(f"  {'':-<10} {'':->9} {'':->9} {'':->10}  {'':->10} {'':->11}")

    variant_means:   dict[str, list[float]] = {}
    variant_medians: dict[str, list[float]] = {}

    for variant in VARIANTS:
        runs    = all_results[variant]
        means   = [r["mean_ep_len"]   for r in runs]
        medians = [r["median_ep_len"] for r in runs]
        variant_means[variant]   = means
        variant_medians[variant] = medians

        mu_m    = float(np.mean(means))
        sig_m   = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        med_m   = float(np.median(means))
        mu_med  = float(np.mean(medians))
        med_med = float(np.median(medians))
        print(f"  {variant:<10} {mu_m:>9.1f} {sig_m:>9.1f} {med_m:>10.1f}"
              f"  {mu_med:>10.1f} {med_med:>11.1f}")

    or_runs = [r for r in all_results["OR"] if "or_sz_mean" in r]
    if or_runs:
        all_sz   = [r["or_sz_mean"]   for r in or_runs]
        all_beta = [r["or_beta_mean"] for r in or_runs]
        print(f"\n  OR params aprendidos (media sobre seeds e passos de avaliacao):")
        print(f"    safety_zone   mu={np.mean(all_sz):.4f}  std={np.std(all_sz):.4f}"
              f"  [min={min(all_sz):.4f}  max={max(all_sz):.4f}]")
        print(f"    penalty_ratio mu={np.mean(all_beta):.4f}  std={np.std(all_beta):.4f}"
              f"  [min={min(all_beta):.4f}  max={max(all_beta):.4f}]")

    statistical_report(variant_means, alpha=args.alpha)

    # ── Gráfico das curvas de aprendizado ─────────────────────────────────────
    plot_learning_curves(
        all_results      = all_results,
        total_timesteps  = timesteps,
        output_path      = args.output,
        smooth_window    = args.smooth_window,
        n_points         = CURVE_POINTS,
    )


if __name__ == "__main__":
    main()
