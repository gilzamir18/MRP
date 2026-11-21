"""
r2_death_cause.py — descriptive follow-up to R2-C (NOT pre-registered; added 2026-11-21 after the
C1/C2 predictions failed).  For the saved R2_term_diag agents (DQ and OR, F1-F6, with termination),
replays the evaluation episodes (same seeds and action RNG as `evaluate`) and records, for each
death, whether the violated variable left the viable range from above (over-consumption, v > +1)
or below (depletion, v < -1).  Depletion only decreases a variable, so a violation with the
variable's previous value > 0 is an over-consumption death.
Output: docs/results_v2/analysis/R2C_death_cause.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from forage_env import ForageEnv, make_reward_fn
from forage_ppo import Agent, agent_fns

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "docs", "results_v2")
OUT = os.path.join(RES, "analysis")


def replay(ag, k, kw, seeds, deterministic):
    act, _ = agent_fns(ag, act_seed=0, deterministic=deterministic)
    env = ForageEnv(len(seeds), k, make_reward_fn("Alive"), terminate=True, **kw)
    for i, s in enumerate(seeds):
        env.reset_one(i, s)
    alive = np.ones(env.E, bool)
    L = np.zeros(env.E, int)
    cause = np.array(["none"] * env.E, dtype=object)
    while alive.any():
        v_before = env.v.copy()
        _, term, trunc, info = env.step(act(env))
        al = alive.copy()
        L += al
        for e in np.where(al & term)[0]:
            j = np.where(info["viol"][e])[0]
            cause[e] = "over" if np.all(v_before[e, j] > 0) else ("depl" if np.all(v_before[e, j] < 0) else "mixed")
        alive &= ~(term | trunc)
    return L, cause


def main():
    torch.set_num_threads(4)
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "R2_term_diag", "*.json"))):
        r = json.load(open(f))
        j, cp = r["job"], r["cond_params"]
        k = cp["k"]
        kw = dict(W=cp["W"], decay_mu=cp["decay_mu"])
        from forage_run import EVAL_SEEDS
        probe = ForageEnv(1, k, make_reward_fn("Alive"), terminate=True, **kw)
        ag = Agent(probe.obs_dim, probe.n_actions)
        ag.load_state_dict(torch.load(f.replace(".json", ".pt")))
        ag.eval()
        ag.sync()
        for pol, det in (("stoch", False), ("det", True)):
            L, cause = replay(ag, k, kw, EVAL_SEEDS, det)
            ref = [e["length"] for e in r["eval" if pol == "stoch" else "eval_det"]]
            rows.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"], policy=pol,
                             matches_eval=bool(np.array_equal(L, ref)), mean_len=L.mean(),
                             n_death=int((cause != "none").sum()), n_over=int((cause == "over").sum()),
                             n_depl=int((cause == "depl").sum()), n_mixed=int((cause == "mixed").sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R2C_death_cause.csv"), index=False)
    g = df.groupby(["policy", "variant", "cond"])[["mean_len", "n_death", "n_over", "n_depl", "n_mixed"]].sum()
    g["over_share"] = g.n_over / g.n_death
    g["mean_len"] = df.groupby(["policy", "variant", "cond"]).mean_len.mean()
    print("replay matches evaluation:", df.matches_eval.mean())
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
