"""
forage_recovery_references.py — recovery test for the two FIXED reference policies
(greedy heuristic and uniform-random policy) in F1--F4.  No training.

Reproduces the protocol of forage_ppo.recovery_test exactly (same evaluation seeds
1000-1019, same deficit -(1 - 25 * decay_mu), zone 0.5, horizon 150, termination on,
one disturbed variable at a time, the others at 0, action RNG default_rng(0) per
disturbed variable), with the learned agent replaced by the reference policy, which
acts on the environment state as in forage_env.run_reference.

Output: docs/results_v2_noadapt/analysis/FORAGE_recovery_references.csv
    one row per (cond, policy): n_episodes (= 20 * k), n_success, success_rate,
    n_died, mean_t_rec (successes only).

    python forage_recovery_references.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from forage_env import ForageEnv, GreedyPolicy, policy_random  # noqa: E402
from forage_run import FCONDS, MAIN_CONDS, EVAL_SEEDS, DEFAULTS  # noqa: E402

OUT = os.path.join(HERE, "..", "docs", "results_v2_noadapt", "analysis", "FORAGE_recovery_references.csv")


def recovery_test_policy(policy, k, cond_kw, seeds, deficit, zone=0.5, horizon=150, act_seed=0):
    """Same loop as forage_ppo.recovery_test, with `policy(env, rng)` instead of ag.np_act."""
    out = []
    for jvar in range(k):
        env = ForageEnv(len(seeds), k, lambda d0, d1, z=None, beta=None: np.zeros(d0.shape[0]),
                        terminate=True, **cond_kw)
        for i, s in enumerate(seeds):
            env.reset_one(i, s)
            env.v[i] = 0.0
            env.v[i, jvar] = deficit
            env.obs_v[i] = env._noisy(i)
        if hasattr(policy, "reset"):
            policy.reset()
        rng = np.random.default_rng(act_seed)
        E = env.E
        done = np.zeros(E, bool)
        success = np.zeros(E, bool)
        t_rec = np.full(E, -1)
        died = np.zeros(E, bool)
        for t in range(1, horizon + 1):
            a = policy(env, rng)
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


def main():
    rows = []
    for cond in MAIN_CONDS:
        c = FCONDS[cond]
        k = c["k"]
        kw = dict(W=DEFAULTS["W"], decay_mu=c["decay_mu"])
        deficit = -(1.0 - 25 * c["decay_mu"])
        g_stale = GreedyPolicy()
        pols = (("greedy", GreedyPolicy()), ("random", policy_random),
                # documentation only: greedy whose target is NOT reset between disturbed variables,
                # so each new episode starts heading for the previous episode's target (not the
                # heuristic as defined; reproduces the older calibration figure 79/80 in F3/F4)
                ("greedy_stale_target", lambda e, r: g_stale(e, r)))
        for name, pol in pols:
            rec = recovery_test_policy(pol, k, kw, EVAL_SEEDS, deficit)
            n = len(rec)
            ns = sum(r["success"] for r in rec)
            nd = sum(r["died"] for r in rec)
            tr = [r["t_rec"] for r in rec if r["success"]]
            rows.append({"cond": cond, "k": k, "decay_mu": c["decay_mu"], "deficit": round(deficit, 4),
                         "policy": name, "n_episodes": n, "n_success": ns,
                         "success_rate": ns / n, "n_died": nd,
                         "mean_t_rec": (sum(tr) / len(tr)) if tr else ""})
            print(cond, name, f"{ns}/{n} = {ns / n:.4f}", "died", nd)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
