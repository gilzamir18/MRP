"""
r2_consume_run.py — descriptive follow-up to R2-C (NOT pre-registered; added 2026-11-21 after the
third review asked whether over-consumption deaths come from a degenerate policy that walks to a
station and keeps consuming).  Replays the evaluation episodes of the saved R2_term_diag agents
(DQ and OR, F1-F6, with termination; same seeds and action RNG as `evaluate`, as in
r2_death_cause.py) and records, for each over-consumption death:
  - run_len : number of consecutive effective consume actions (at the station of the violated
              variable) that end at death;
  - v_start : value of that variable at the start of the run (gain 0.2 per consume);
  - n_vars  : number of distinct variables consumed at least once during the episode;
  - n_eat   : total number of effective consume actions in the episode.
Output: docs/results_v2/analysis/R2C_consume_run.csv (one row per over-consumption death) and
        R2C_consume_run_summary.csv (per policy x variant x condition).
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
    E = env.E
    alive = np.ones(E, bool)
    L = np.zeros(E, int)
    run = np.zeros((E, k), int)          # current run of consecutive effective consumes per variable
    vstart = np.zeros((E, k))
    ever = np.zeros((E, k), bool)
    n_eat = np.zeros(E, int)
    deaths = []
    while alive.any():
        v_before = env.v.copy()
        _, term, trunc, info = env.step(act(env))
        al = alive.copy()
        L += al
        eat = info["eat"]
        # a run continues only if the same variable is consumed on consecutive steps
        new = eat & (run == 0)
        vstart[new] = v_before[new]
        run = np.where(eat, run + 1, 0)
        ever |= eat & al[:, None]
        n_eat += (eat.any(1) & al)
        for e in np.where(al & term)[0]:
            j = np.where(info["viol"][e])[0]
            if np.all(v_before[e, j] > 0):            # over-consumption death (as in r2_death_cause)
                jj = j[0]
                deaths.append(dict(ep=e, L=int(L[e]), run_len=int(run[e, jj]), v_start=float(vstart[e, jj]),
                                   n_vars=int(ever[e].sum()), n_eat=int(n_eat[e])))
        alive &= ~(term | trunc)
    return L, deaths


def main():
    torch.set_num_threads(4)
    from forage_run import EVAL_SEEDS
    rows, check = [], []
    for f in sorted(glob.glob(os.path.join(RES, "R2_term_diag", "*.json"))):
        r = json.load(open(f))
        j, cp = r["job"], r["cond_params"]
        k = cp["k"]
        kw = dict(W=cp["W"], decay_mu=cp["decay_mu"])
        probe = ForageEnv(1, k, make_reward_fn("Alive"), terminate=True, **kw)
        ag = Agent(probe.obs_dim, probe.n_actions)
        ag.load_state_dict(torch.load(f.replace(".json", ".pt")))
        ag.eval()
        ag.sync()
        for pol, det in (("stoch", False), ("det", True)):
            L, deaths = replay(ag, k, kw, EVAL_SEEDS, det)
            ref = [e["length"] for e in r["eval" if pol == "stoch" else "eval_det"]]
            check.append(bool(np.array_equal(L, ref)))
            for d in deaths:
                rows.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"], policy=pol, k=k, **d))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R2C_consume_run.csv"), index=False)
    print("replay matches evaluation:", np.mean(check), f"({len(check)} replays)")
    g = df.groupby(["policy", "variant", "cond"])
    s = g.agg(n_over=("run_len", "size"), run_med=("run_len", "median"), run_min=("run_len", "min"),
              run_max=("run_len", "max"), run_mean=("run_len", "mean"),
              vstart_mean=("v_start", "mean"), vstart_neg=("v_start", lambda x: (x < 0).mean()),
              nvars_mean=("n_vars", "mean"), nvars_eq_k=("n_vars", lambda x: np.nan),
              L_mean=("L", "mean"), neat_mean=("n_eat", "mean"))
    kk = g.k.first()
    s["share_all_vars"] = g.apply(lambda x: (x.n_vars == x.k).mean())
    s["share_one_var"] = g.apply(lambda x: (x.n_vars == 1).mean())
    s = s.drop(columns="nvars_eq_k")
    s["k"] = kk
    s.to_csv(os.path.join(OUT, "R2C_consume_run_summary.csv"))
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
