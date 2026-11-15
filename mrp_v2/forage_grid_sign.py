"""
forage_grid_sign.py — does the sign/informativeness of OR(z, beta) explain the fitness landscape?

For each condition, transitions (d0, d1) are collected with a uniformly random policy on the
no-termination foraging task (identical trajectories for every grid point).  For each (z, beta)
of the F_zbgrid study we compute the mean per-step OR reward and the fraction of steps on which
the reward equals +1 (saturation, i.e. no information).  Output: FORAGE_Q4_grid_sign.csv and
a landscape figure with the zero-mean-reward contour.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import OUT, FIG  # noqa: E402
from forage_env import ForageEnv, make_reward_fn, or_per_variable, policy_random  # noqa: E402
from forage_run import FCONDS, Z_GRID, B_GRID, EVAL_SEEDS  # noqa: E402

CONDS = ["F1", "F2", "F3", "F4"]


def transitions(cond, n_rounds=10):
    """Transitions experienced by a random (initial) policy WITH termination, up to and including
    the terminal step -- the experience on which early learning, and the sign effect, act."""
    c = FCONDS[cond]
    D0, D1 = [], []
    rng = np.random.default_rng(0)
    for rnd in range(n_rounds):
        seeds = [s + 1000 * rnd for s in EVAL_SEEDS]
        env = ForageEnv(len(seeds), c["k"], make_reward_fn("Alive"), terminate=True, W=9, decay_mu=c["decay_mu"])
        for i, s in enumerate(seeds):
            env.reset_one(i, s)
        alive = np.ones(env.E, bool)
        while alive.any():
            _, term, trunc, info = env.step(policy_random(env, rng))
            D0.append(info["dev_prev"][alive])
            D1.append(info["dev"][alive])
            alive &= ~(term | trunc)
    return np.concatenate(D0), np.concatenate(D1)


def main():
    rows = []
    for cond in CONDS:
        d0, d1 = transitions(cond)
        for z in Z_GRID:
            for b in B_GRID:
                r = or_per_variable(d0, d1, z, b).mean(1)
                rows.append(dict(cond=cond, z=z, beta=b, mean_r=float(r.mean()),
                                 frac_pos=float((r > 0).mean()), saturated=float((r >= 1.0 - 1e-12).mean())))
    S = pd.DataFrame(rows)
    M = pd.read_csv(os.path.join(OUT, "FORAGE_Q4_zbgrid_means.csv"))
    J = S.merge(M, on=["cond", "z", "beta"])
    J.to_csv(os.path.join(OUT, "FORAGE_Q4_grid_sign.csv"), index=False)
    for cond in CONDS:
        j = J[J.cond == cond]
        neg = j[j.mean_r < 0]["mean"]
        pos = j[j.mean_r >= 0]
        print(cond, "cells with mean r < 0:", len(neg), "RMST", neg.round(0).tolist())
        print("   rho(saturation, RMST | mean r >= 0) =",
              round(pos[["saturated", "mean"]].corr(method="spearman").iloc[0, 1], 2),
              " n =", len(pos))
    # figure
    fig, axes = plt.subplots(1, len(CONDS), figsize=(4.3 * len(CONDS), 3.9))
    for ax, cond in zip(axes, CONDS):
        j = J[J.cond == cond]
        piv = j.pivot(index="z", columns="beta", values="mean")
        mr = j.pivot(index="z", columns="beta", values="mean_r")
        im = ax.imshow(piv.values, origin="lower", cmap="viridis", aspect="auto")
        ax.contour(mr.values, levels=[0.0], colors="w", linewidths=2, origin="lower")
        for (i, k), val in np.ndenumerate(piv.values):
            ax.text(k, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    color="w" if val < np.nanmax(piv.values) * 0.6 else "k")
        ax.plot([B_GRID.index(1.0)], [Z_GRID.index(0.5)], marker="s", mfc="none", mec="r", ms=16, mew=1.5)
        ax.set_xticks(range(len(B_GRID)))
        ax.set_xticklabels([f"{b:g}" for b in B_GRID])
        ax.set_yticks(range(len(Z_GRID)))
        ax.set_yticklabels([f"{z:g}" for z in Z_GRID])
        ax.set_xlabel(r"$\beta$")
        k, d = FCONDS[cond]["k"], FCONDS[cond]["decay_mu"]
        ax.set_title(f"{cond} (k={k}, depletion {d})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel(r"$z_{\mathrm{safe}}$")
    fig.suptitle("Fixed OR, foraging with termination: RMST (τ = 500) over (z_safe, β); white line: mean per-step "
                 "reward = 0 (below/right: negative); red square: default (0.5, 1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FORAGE_Q4_zbgrid.png"), dpi=160)
    plt.close(fig)
    print("saved figure")


if __name__ == "__main__":
    main()
