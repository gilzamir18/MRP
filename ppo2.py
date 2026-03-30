"""
ppo.py — Proximal Policy Optimization (from scratch) for HomeostaticEnv.

Style mirrors reinforce.py: same run_experiment / plot / main structure,
but uses the full PPO toolkit:
  • Actor-Critic shared trunk                (separate value head)
  • Generalized Advantage Estimation (GAE)
  • Clipped surrogate objective              (PPO-Clip)
  • Value function loss (MSE, optionally clipped)
  • Entropy bonus
  • Gradient norm clipping
  • Multiple update epochs per rollout
  • Mini-batch SGD within each update

CLI is modelled after stable-baselines3 PPO hyperparameter names.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

from env import HomeostaticEnv

# ── plotting style ────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Actor-Critic Network
# ═════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Shared MLP trunk → separate Actor (mean) and Critic (value) heads.

    Actor outputs Gaussian parameters for continuous actions.
    log_std is a learned parameter vector (state-independent), matching the
    SB3 MlpPolicy default.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head: outputs mean of the action distribution
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head: outputs scalar state value V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Learned log-std (state-independent), initialised to 0 → std = 1
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init (recommended for PPO)."""
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
        mean      = self.actor_head(features)
        value     = self.critic_head(features).squeeze(-1)
        std       = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action and return (action, log_prob, value)."""
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Re-evaluate stored actions for the update step."""
        mean, std, value = self(obs)
        dist     = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# ═════════════════════════════════════════════════════════════════════════════
# 2.  GAE – Generalised Advantage Estimation
# ═════════════════════════════════════════════════════════════════════════════

def compute_gae(
    rewards:    list[float],
    values:     list[float],
    dones:      list[bool],
    last_value: float,
    gamma:      float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """
    Returns (advantages, returns) as plain Python lists.
    last_value: V(s_{T+1}) – the bootstrap value at the end of the rollout.
    """
    advantages = []
    gae = 0.0
    next_value = last_value
    next_done  = False

    for t in reversed(range(len(rewards))):
        mask   = 1.0 - float(dones[t])
        delta  = rewards[t] + gamma * next_value * mask - values[t]
        gae    = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
        next_done  = dones[t]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Rollout buffer
# ═════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores one rollout worth of data and converts to tensors on demand."""

    def __init__(self):
        self.obs:       list = []
        self.actions:   list = []
        self.log_probs: list = []
        self.values:    list = []
        self.rewards:   list = []
        self.dones:     list = []

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
        obs_t       = torch.FloatTensor(np.array(self.obs)).to(device)
        actions_t   = torch.FloatTensor(np.array(self.actions)).to(device)
        log_probs_t = torch.FloatTensor(self.log_probs).to(device)
        advs_t      = torch.FloatTensor(advantages).to(device)
        returns_t   = torch.FloatTensor(returns).to(device)
        return obs_t, actions_t, log_probs_t, advs_t, returns_t

    def __len__(self):
        return len(self.rewards)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  PPO update
# ═════════════════════════════════════════════════════════════════════════════

def ppo_update(
    ac:           ActorCritic,
    optimizer:    optim.Optimizer,
    obs_t:        torch.Tensor,
    actions_t:    torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages_t: torch.Tensor,
    returns_t:    torch.Tensor,
    old_values_t: torch.Tensor,
    n_epochs:     int   = 10,
    batch_size:   int   = 64,
    clip_range:   float = 0.2,
    vf_coef:      float = 0.5,
    ent_coef:     float = 0.0,
    max_grad_norm: float = 0.5,
    clip_vf_loss: bool  = True,
) -> dict:
    """Runs n_epochs of mini-batch PPO updates. Returns mean losses."""
    n = obs_t.shape[0]
    metrics = {"pg_loss": [], "vf_loss": [], "entropy": [], "clip_frac": []}

    for _ in range(n_epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]

            mb_obs      = obs_t[idx]
            mb_actions  = actions_t[idx]
            mb_old_lp   = old_log_probs[idx]
            mb_adv      = advantages_t[idx]
            mb_returns  = returns_t[idx]
            mb_old_val  = old_values_t[idx]

            # Normalise advantages within mini-batch
            if mb_adv.shape[0] > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            new_log_probs, entropy, new_values = ac.evaluate(mb_obs, mb_actions)

            # ── Policy loss (PPO-Clip) ───────────────────────────────────────
            ratio      = torch.exp(new_log_probs - mb_old_lp)
            pg_loss1   = -mb_adv * ratio
            pg_loss2   = -mb_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            pg_loss    = torch.max(pg_loss1, pg_loss2).mean()
            clip_frac  = ((ratio - 1).abs() > clip_range).float().mean().item()

            # ── Value loss (optionally clipped, like SB3) ────────────────────
            if clip_vf_loss:
                vf_loss_unclipped = (new_values - mb_returns) ** 2
                vf_clipped        = mb_old_val + torch.clamp(
                    new_values - mb_old_val, -clip_range, clip_range
                )
                vf_loss_clipped   = (vf_clipped - mb_returns) ** 2
                vf_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
            else:
                vf_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

            # ── Entropy bonus ────────────────────────────────────────────────
            entropy_loss = -entropy.mean()

            loss = pg_loss + vf_coef * vf_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            optimizer.step()

            metrics["pg_loss"].append(pg_loss.item())
            metrics["vf_loss"].append(vf_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["clip_frac"].append(clip_frac)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Main training loop
# ═════════════════════════════════════════════════════════════════════════════

def train_ppo(
    env,
    ac:             ActorCritic,
    optimizer:      optim.Optimizer,
    total_timesteps: int   = 200_000,
    n_steps:         int   = 2048,   # rollout length per update
    n_epochs:        int   = 10,
    batch_size:      int   = 64,
    gamma:           float = 0.99,
    gae_lambda:      float = 0.95,
    clip_range:      float = 0.2,
    vf_coef:         float = 0.5,
    ent_coef:        float = 0.0,
    max_grad_norm:   float = 0.5,
    clip_vf_loss:    bool  = True,
    device:          str   = "cpu",
) -> dict:
    """
    Collects rollouts of length `n_steps`, computes GAE, then runs PPO updates.
    Returns a metrics dict with one entry per update iteration.
    """
    log = {
        "update":       [],
        "timestep":     [],
        "ep_len_mean":  [],
        "ep_rew_mean":  [],
        "pg_loss":      [],
        "vf_loss":      [],
        "entropy":      [],
        "clip_frac":    [],
    }

    buffer   = RolloutBuffer()
    obs, _   = env.reset()
    done     = False
    truncated = False

    total_steps  = 0
    update_count = 0

    # Running episode bookkeeping
    ep_rewards: list[float] = []
    ep_lengths: list[int]   = []
    current_ep_reward = 0.0
    current_ep_len    = 0

    pbar = tqdm(total=total_timesteps, desc="PPO Training", leave=False, unit="step")

    while total_steps < total_timesteps:
        buffer.clear()

        # ── Collect n_steps of experience ────────────────────────────────────
        for _ in range(n_steps):
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, value = ac.get_action(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, _ = env.step(action_np)

            buffer.add(
                obs      = obs,
                action   = action_np,
                log_prob = log_prob.item(),
                value    = value.item(),
                reward   = float(reward),
                done     = bool(done or truncated),
            )

            current_ep_reward += float(reward)
            current_ep_len    += 1
            total_steps       += 1
            pbar.update(1)

            if done or truncated:
                ep_rewards.append(current_ep_reward)
                ep_lengths.append(current_ep_len)
                current_ep_reward = 0.0
                current_ep_len    = 0
                obs, _ = env.reset()
                done = truncated = False
            else:
                obs = next_obs

            if total_steps >= total_timesteps:
                break

        # ── Bootstrap last value ──────────────────────────────────────────────
        with torch.no_grad():
            last_obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(device)
            _, _, last_v = ac.get_action(last_obs_t)
            last_value   = last_v.item() if not (done or truncated) else 0.0

        # ── GAE ───────────────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            last_value, gamma, gae_lambda
        )

        obs_t, actions_t, old_log_probs_t, adv_t, returns_t = buffer.to_tensors(
            advantages, returns, device=device
        )
        old_values_t = torch.FloatTensor(buffer.values).to(device)

        # ── PPO update ────────────────────────────────────────────────────────
        update_metrics = ppo_update(
            ac, optimizer,
            obs_t, actions_t, old_log_probs_t,
            adv_t, returns_t, old_values_t,
            n_epochs=n_epochs, batch_size=batch_size,
            clip_range=clip_range, vf_coef=vf_coef,
            ent_coef=ent_coef, max_grad_norm=max_grad_norm,
            clip_vf_loss=clip_vf_loss,
        )

        update_count += 1

        mean_ep_len = float(np.mean(ep_lengths[-20:])) if ep_lengths else 0.0
        mean_ep_rew = float(np.mean(ep_rewards[-20:])) if ep_rewards else 0.0

        log["update"].append(update_count)
        log["timestep"].append(total_steps)
        log["ep_len_mean"].append(mean_ep_len)
        log["ep_rew_mean"].append(mean_ep_rew)
        log["pg_loss"].append(update_metrics["pg_loss"])
        log["vf_loss"].append(update_metrics["vf_loss"])
        log["entropy"].append(update_metrics["entropy"])
        log["clip_frac"].append(update_metrics["clip_frac"])

    pbar.close()
    return log


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Experiment runner  (mirrors reinforce.py style)
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(
    rewarding_variant: str,
    seed:              int,
    num_vars:          int   = 100,
    max_steps:         int   = 1000,
    total_timesteps:   int   = 200_000,
    safety_zone:       float = 0.5,
    penalty_ratio:     float = 2.0,   # β: -βα vs +α  (2.0 = paper default)
    reward_scale:      float = 1.0,   # α magnitude (scales all OR rewards)
    n_eval_episodes:   int   = 30,    # evaluation rollouts after training
    obs_noise_std:     float = 0.0,
    decay_std:        float = 0.001,
    decay_ratio_mu:     float = 0.001,
    correction_budget: float = 1.0,
    # PPO hyperparameters (SB3-style names)
    learning_rate:     float = 3e-4,
    n_steps:           int   = 2048,
    batch_size:        int   = 64,
    n_epochs:          int   = 10,
    gamma:             float = 0.99,
    gae_lambda:        float = 0.95,
    clip_range:        float = 0.2,
    vf_coef:           float = 0.5,
    ent_coef:          float = 0.0,
    max_grad_norm:     float = 0.5,
    hidden_dim:        int   = 64,
    device:            str   = "cpu",
) -> tuple[pd.DataFrame, dict]:

    print(f"Starting PPO  {rewarding_variant} – Seed {seed}  β={penalty_ratio}  sz={safety_zone}  α={reward_scale}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env_kwargs = dict(
        num_vars           = num_vars,
        rewarding          = rewarding_variant,
        reward_scale       = reward_scale,
        max_steps          = max_steps,
        safety_zone        = safety_zone,
        penalty_ratio      = penalty_ratio,
        obs_noise_std      = obs_noise_std,
        decay_std          = decay_std,
        decay_ratio_mu     = decay_ratio_mu,
        correction_budget  = correction_budget,
        use_learned_params = False,
    )

    env = HomeostaticEnv(**env_kwargs)
    env.reset(seed=seed)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ac        = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=learning_rate, eps=1e-5)

    log = train_ppo(
        env, ac, optimizer,
        total_timesteps  = total_timesteps,
        n_steps          = n_steps,
        n_epochs         = n_epochs,
        batch_size       = batch_size,
        gamma            = gamma,
        gae_lambda       = gae_lambda,
        clip_range       = clip_range,
        vf_coef          = vf_coef,
        ent_coef         = ent_coef,
        max_grad_norm    = max_grad_norm,
        device           = device,
    )
    env.close()

    # ── Dedicated evaluation (stochastic policy, fresh env) ──────────────────
    eval_env = HomeostaticEnv(**env_kwargs)
    ac.eval()
    eval_ep_lens  = []
    eval_ep_rews  = []

    with torch.no_grad():
        for _ in range(n_eval_episodes):
            obs_e, _ = eval_env.reset()
            ep_len, ep_rew = 0, 0.0
            d = tr = False
            while not (d or tr):
                ot = torch.FloatTensor(obs_e).unsqueeze(0).to(device)                
                act, _, _ = ac.get_action(ot, deterministic=False)
                obs_e, r, d, tr, _ = eval_env.step(act.squeeze(0).cpu().numpy())
                ep_len += 1
                ep_rew += float(r)
            eval_ep_lens.append(ep_len)
            eval_ep_rews.append(ep_rew)
    eval_env.close()
    ac.train()

    df = pd.DataFrame(log)
    df["variant"] = rewarding_variant
    df["seed"]    = seed

    eval_window = max(1, len(df) // 5)
    final_result = {
        "variant":                    rewarding_variant,
        "seed":                       seed,
        "penalty_ratio":              penalty_ratio,
        # training-log estimates (last 20%)
        "train_mean_ep_length":       df["ep_len_mean"].tail(eval_window).mean(),
        "train_mean_ep_reward":       df["ep_rew_mean"].tail(eval_window).mean(),
        "final_clip_frac":            df["clip_frac"].tail(eval_window).mean(),
        # dedicated evaluation (mean + std → cycling indicator)
        "final_mean_episode_length":  float(np.mean(eval_ep_lens)),
        "final_std_episode_length":   float(np.std(eval_ep_lens, ddof=1)) if len(eval_ep_lens) > 1 else 0.0,
        "final_cv_episode_length":    float(np.std(eval_ep_lens, ddof=1) / (np.mean(eval_ep_lens) + 1e-9)),
        "final_mean_episode_reward":  float(np.mean(eval_ep_rews)),
        "final_std_episode_reward":   float(np.std(eval_ep_rews, ddof=1)) if len(eval_ep_rews) > 1 else 0.0,
    }
    return df, final_result


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Plots  (same structure as reinforce.py)
# ═════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(df_all: pd.DataFrame, base_dir: str = "figures"):
    os.makedirs(base_dir, exist_ok=True)

    metrics = [
        ("ep_len_mean",  "Mean Episode Length"),
        ("ep_rew_mean",  "Mean Episode Reward"),
        ("entropy",      "Policy Entropy"),
        ("clip_frac",    "PPO Clip Fraction"),
    ]

    for col, title in metrics:
        smooth_col = f"_smooth_{col}"
        df_all[smooth_col] = df_all.groupby(["variant", "seed"])[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=df_all, x="timestep", y=smooth_col,
            hue="variant", errorbar="sd",
        )
        plt.title(f"PPO – {title}")
        plt.xlabel("Timesteps")
        plt.ylabel(title)
        plt.legend(title="Variant")
        plt.tight_layout()
        fname = col.replace("_", "")
        plt.savefig(os.path.join(base_dir, f"ppo_curve_{fname}.png"))
        plt.savefig(os.path.join(base_dir, f"ppo_curve_{fname}.pdf"))
        plt.close()


def plot_final_metrics(df_results: pd.DataFrame, base_dir: str = "figures"):
    os.makedirs(base_dir, exist_ok=True)

    metrics = [
        ("final_mean_episode_length", "Final Episode Length"),
        ("final_mean_episode_reward", "Final Episode Reward"),
    ]

    for col, title in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_results,  x="variant", y=col)
        sns.stripplot(data=df_results, x="variant", y=col, color="black", alpha=0.5)
        plt.title(f"PPO – {title}")
        plt.xlabel("Variant")
        plt.ylabel(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = col.replace("final_mean_episode_", "ppo_box_")
        plt.savefig(os.path.join(base_dir, f"{fname}.png"))
        plt.savefig(os.path.join(base_dir, f"{fname}.pdf"))
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CLI entry-point  (SB3-style hyperparameter names)
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PPO Ablation Study on HomeostaticEnv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument("--num-vars",     type=int,   default=100,     help="State dimensionality")
    parser.add_argument("--max-steps",    type=int,   default=1000,    help="Max steps per episode")
    parser.add_argument("--seeds",        type=int,   default=5,       help="Number of seeds per variant")

    # Training
    parser.add_argument("--timesteps",    type=int,   default=200_000, help="Total env timesteps")
    parser.add_argument("--test-mode",    action="store_true",         help="Quick smoke-test mode")

    # PPO hyperparameters (SB3 names)
    parser.add_argument("--learning-rate", type=float, default=3e-4,  help="Adam learning rate")
    parser.add_argument("--n-steps",       type=int,   default=2048,  help="Rollout length per update")
    parser.add_argument("--batch-size",    type=int,   default=64,    help="Mini-batch size")
    parser.add_argument("--n-epochs",      type=int,   default=10,    help="Update epochs per rollout")
    parser.add_argument("--gamma",         type=float, default=0.99,  help="Discount factor")
    parser.add_argument("--gae-lambda",    type=float, default=0.95,  help="GAE lambda")
    parser.add_argument("--clip-range",    type=float, default=0.2,   help="PPO clip epsilon")
    parser.add_argument("--vf-coef",       type=float, default=0.5,   help="Value function loss coeff")
    parser.add_argument("--ent-coef",      type=float, default=0.0,   help="Entropy bonus coeff")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,   help="Gradient clipping norm")
    parser.add_argument("--hidden-dim",    type=int,   default=64,    help="Hidden layer size")
    parser.add_argument("--device",        type=str,   default="cpu", help="Torch device")

    args = parser.parse_args()

    variants    = [args.rewarding] if args.rewarding else ["QD", "operational_regimes", "default"]
    seeds_list  = list(range(42, 42 + args.seeds))
    timesteps   = args.timesteps

    if args.test_mode:
        print("TEST MODE ACTIVATED")
        variants   = [args.rewarding] if args.rewarding else ["QD", "operational_regimes"]
        seeds_list = [42, 43]
        timesteps  = 20_000

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    all_progress  = []
    all_results   = []

    for fns in variants:
        for seed in seeds_list:
            df_prog, res = run_experiment(
                reward_fns        = fns,
                seed              = seed,
                num_vars          = args.num_vars,
                max_steps         = args.max_steps,
                total_timesteps   = timesteps,
                learning_rate     = args.learning_rate,
                n_steps           = args.n_steps,
                batch_size        = args.batch_size,
                n_epochs          = args.n_epochs,
                gamma             = args.gamma,
                gae_lambda        = args.gae_lambda,
                clip_range        = args.clip_range,
                vf_coef           = args.vf_coef,
                ent_coef          = args.ent_coef,
                max_grad_norm     = args.max_grad_norm,
                hidden_dim        = args.hidden_dim,
                device            = args.device,
            )
            all_progress.append(df_prog)
            all_results.append(res)

    df_all_progress = pd.concat(all_progress, ignore_index=True)
    df_all_results  = pd.DataFrame(all_results)

    print("\nSaving metrics and summary tables...")
    df_all_progress.to_csv("results/ppo_training_log.csv",    index=False)
    df_all_results.to_csv("results/ppo_final_results.csv",    index=False)

    print("Generating plots...")
    plot_learning_curves(df_all_progress, base_dir="figures")
    plot_final_metrics(df_all_results,    base_dir="figures")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = df_all_results.groupby("variant").agg(
        Ep_Len_Mean = ("final_mean_episode_length", "mean"),
        Ep_Len_Std  = ("final_mean_episode_length", "std"),
        Ep_Rew_Mean = ("final_mean_episode_reward",  "mean"),
        Ep_Rew_Std  = ("final_mean_episode_reward",  "std"),
    ).reset_index()

    print("\n### PPO Ablation Summary ###\n")
    print(summary.to_string(index=False))
    print("\nFiles saved in ./results/ and ./figures/")


if __name__ == "__main__":
    main()
