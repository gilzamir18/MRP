# When to Act: Regime-Based Reward Design for Adaptive Homeostatic Behavior

> Code repository for the paper *"When to Act: Regime-Based Reward Design for Adaptive Homeostatic Behavior"*.

---

## Overview

This repository implements **Operational Regimes (OR)**, a heuristic reward formulation for homeostatic reinforcement learning that structures regulation into three discrete regimes:

| Regime | Condition | Per-variable reward |
|---|---|---|
| **Maintenance** | variable inside safety zone | `+1` |
| **Recovery** | outside safety zone, condition improving | `+1` |
| **Non-recovery** | outside safety zone, condition not improving | `−β` |

Unlike continuous baselines (Standard Homeostatic / Differential Quadratic), OR uses **binary per-variable improvement indicators**, making it robust to observation noise and efficient under corrective budget scarcity.  Both the safety zone size `z_safe` and the penalty parameter `β` are **learned jointly with the policy** via PPO-Clip policy gradient — no manual tuning required.

---

## Repository structure

```
.
├── env.py                    # HomeostaticEnv (Multi-Resource Problem, MRP)
├── ppo.py                    # Custom PPO implementation (Actor-Critic from scratch)
├── ppo2.py                   # Alternative PPO variant
├── multivariant_test_ppo.py  # Factorial experiment (C1–C8) + learning-curve plots
├── requirements.txt          # Python dependencies
├── docs/                     # Figures, results, logs, bib (not versioned)
└── ...
```

---

## The Multi-Resource Problem (MRP)

`HomeostaticEnv` (in `env.py`) is a Gymnasium environment where the agent simultaneously regulates `n` independent homeostatic variables.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `num_vars` | `2` | Number of homeostatic variables (`n`) |
| `rewarding` | `"default"` | Reward function: `"default"` (DR), `"QD"`, `"OR"`, or plugin `module:func` (e.g. `rewards:custom_reward`) |
| `correction_budget` | `1.0` | Maximum total correction per step (`b`) |
| `random_budget` | `False` | Sample `b ~ U(0,1]` at each reset |
| `obs_noise_std` | `0.0` | Standard deviation of observation noise (`σ_obs`) |
| `safety_zone` | `0.5` | Safety zone half-width (`z_safe`); `≤0` randomises per variable |
| `penalty_ratio` | `3.0` | Non-recovery penalty `β` |
| `use_learned_params` | `False` | Accept `learned_params` dict from a joint-learning policy |
| `decay_ratio_mu` | `0.001` | Mean decay rate (`μ_δ`) |
| `decay_std` | `0.001` | Decay standard deviation (`σ_δ`) |
| `max_steps` | `1000` | Maximum steps per episode (`T_max`) |

**Observation space:** `4n + 1` continuous values — state variables, survival bounds, safety zones, decay rates, and available budget.

**Action space:** `[0, 1]^n` — fraction of budget allocated to each variable (softmax-normalised internally).

**Termination:** any variable crosses its survival bound `±1.0`, or `T_max` is reached.

**Episode length** is the primary performance metric.

---

## Reward formulations

### Standard Homeostatic (DR / Default)
```
R_t = -Σ (v_i - v_T)²
```

### Differential Quadratic (QD)
```
R_t = D_t − D_{t+1},   where D_t = Σ (v_i − v_T)²
```

### Operational Regimes (OR)
```
r_i = u_i + (1 − u_i)(1 − (1 + β) · m_i)

R_t = (1/n) Σ r_i

where:
  u_i = 1  if  v_i ∈ Z_i  (inside safety zone)
  m_i = 1  if  D_{t+1}_i ≥ D_t_i  (no improvement)
```

Total reward stays in `[−β, +1]` regardless of dimensionality.

---

## Joint learning of OR hyperparameters

When `use_learned_params=True`, the Actor-Critic network is extended with four trainable scalar parameters `{μ_sz, log σ_sz, μ_β, log σ_β}` that define distributions over `z_safe` and `β`.  At each step, values are sampled and their log-probabilities are included in the PPO-Clip objective, so both parameters are updated by the same policy gradient as the control policy.

Learned parameter ranges:
- `z_safe = σ(μ_sz) · 0.8 + 0.1  ∈ (0.1, 0.9)`
- `β = log(1 + exp(μ_β))  ∈ (0, +∞)`

---

## Experimental conditions

The main experiment (`multivariant_test_ppo.py`) uses a **2 × 2 × 2 factorial design**:

| Factor | Levels |
|---|---|
| Dimensionality `n` | 20, 50 |
| Observation noise `σ_obs` | 0.0, 0.2 |
| Correction budget `b` | 0.1, 1.0 |

This yields 8 conditions (C1–C8) × 3 reward functions × 20 seeds = **480 experimental runs**.

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `gymnasium`, `stable-baselines3`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `pandas`.

---

## Usage

![Python package](https://github.com/<your-user>/<your-repo>/actions/workflows/python-package.yml/badge.svg)

### Run the factorial experiment (C1–C8)

```bash
python multivariant_test_ppo.py
```

Additional options:

```bash
# Fewer seeds and evaluation episodes (quick test)
python multivariant_test_ppo.py --train-seeds 5 --eval-episodes 30

# Fast smoke test
python multivariant_test_ppo.py --test-mode

# Regenerate plots from saved results
python multivariant_test_ppo.py --plot-only --results-file docs/results/curves.npy
```

### Quick environment check

```python
from env import HomeostaticEnv

env = HomeostaticEnv(num_vars=20, rewarding="OR", random_budget=True)
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Plugin de recompensas

```python
from env import HomeostaticEnv

# Carrega custom reward em rewards.py
env = HomeostaticEnv(num_vars=10, rewarding="rewards:custom_reward")
obs, info = env.reset(seed=123)

# Também permite atalho se o módulo for rewards
env2 = HomeostaticEnv(num_vars=10, rewarding="custom_reward")
```

---

## Key results (from the paper)

| Condition | OR µ (steps) | QD µ | Default µ | Best |
|---|---|---|---|---|
| C1: n=20, no noise, b=1.0 | 1001 | 1001 | 1001 | n.s. |
| C3: n=20, no noise, b=0.1 | **92.7** | 88.5 | 91.0 | OR |
| C5: n=50, no noise, b=1.0 | **694.6** | 470.7 | 314.4 | OR > QD > Default |
| C6: n=50, noise, b=1.0 | **708.6** | 326.9 | 563.3 | OR > Default > QD |
| C8: n=50, noise, b=0.1 | 45.6 | 44.4 | 44.9 | n.s. |

The most informative result is **C6**: under high dimensionality + observation noise, DQ regresses *below* the standard baseline, while OR maintains stable performance — a consequence of OR's binary per-variable improvement indicators being immune to noise contamination of the continuous difference `D_t − D_{t+1}`.

---

## Citation

```bibtex
@article{or_homeostatic_2025,
  title   = {When to Act: Regime-Based Reward Design for Adaptive Homeostatic Behavior},
  journal = {Journal Title},
  year    = {2025},
  note    = {DOI: 10.1177/ToBeAssigned}
}
```

---

## License

See repository license file.
