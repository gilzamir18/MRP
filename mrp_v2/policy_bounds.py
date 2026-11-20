"""
policy_bounds.py — survival of non-learning reference policies in every condition,
on the SAME evaluation episodes (seeds 1000-1019) used for the trained agents.

Policies
  zero      logits = 0 (uniform allocation b/n to every variable)
  random    logits ~ N(0,1) each step
  prop      allocation proportional to observed |v|
  greedy    near-all budget to the largest observed |v|
  oracle    myopic water-filling on the TRUE state and decay: chooses the allocation
            that minimises max_i |v_i(t+1)|  (an upper reference for what a reactive
            controller can achieve; it is not guaranteed optimal over the horizon)

The point is to establish the attainable range per condition: if the oracle cannot
survive, a condition measures environment infeasibility, not reward design.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mrp_env import MRPEnv, make_reward_fn  # noqa: E402
from runner import CONDITIONS  # noqa: E402

EVAL_SEEDS = list(range(1000, 1020))


def waterfill(e, b):
    """a >= 0, sum a = b, minimising max(e - a). Returns a."""
    e = np.maximum(e, 0.0)
    if e.sum() <= b:                       # can push everything to (at most) zero ...
        a = e.copy()
        a += (b - e.sum()) / len(e)        # ... spend the remainder uniformly
        return a
    s = np.sort(e)[::-1]
    cs = np.cumsum(s)
    k = np.arange(1, len(s) + 1)
    L = (cs - b) / k
    j = np.max(np.where(s - L > 0)[0])
    lvl = L[j]
    return np.maximum(e - lvl, 0.0)


def policy_logits(name, env, rng):
    n = env.n
    if name == "zero":
        return np.zeros(n)
    if name == "random":
        return rng.normal(0, 1, n)
    if name == "prop":
        return np.log(np.abs(env._obs_v) + 1e-6)
    if name == "greedy":
        return 50.0 * np.abs(env._obs_v)
    if name == "oracle":
        e = np.abs(env.v) + env.decay * (np.abs(env.v) > 1e-3)
        a = waterfill(e, env.budget)
        return np.log(a / env.budget + 1e-12)
    raise ValueError(name)


def run(conds=None):
    rfn = make_reward_fn("DR")
    rows = []
    for c, p in (conds or CONDITIONS).items():
        for pol in ("zero", "random", "prop", "greedy", "oracle"):
            env = MRPEnv(p["n"], p["budget"], p["noise"], rfn)
            rng = np.random.default_rng(0)
            for s in EVAL_SEEDS:
                env.reset(s)
                done, L = False, 0
                while not done:
                    _, _, term, trunc, _ = env.step(policy_logits(pol, env, rng))
                    L += 1
                    done = term or trunc
                rows.append(dict(cond=c, n=p["n"], noise=p["noise"], budget=p["budget"],
                                 policy=pol, eval_seed=s, length=L, event=int(term)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = os.path.join(os.environ.get("MRP_RES", os.path.join(HERE, "..", "docs", "results_v2")), "analysis")
    os.makedirs(out, exist_ok=True)
    df = run()
    df.to_csv(os.path.join(out, "policy_bounds_episodes.csv"), index=False)
    tab = df.groupby(["cond", "policy"]).length.mean().unstack()[["zero", "random", "prop", "greedy", "oracle"]]
    tab.to_csv(os.path.join(out, "policy_bounds.csv"))
    print(tab.round(1).to_string())
