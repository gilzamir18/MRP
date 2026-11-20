"""
mrp_v3_check.py — check that the v3 dynamics remove the absorbing state of the MRP.

For every condition and both dynamics ("legacy", "v3"), runs the random-logit policy and the
uniform policy (logits = 0) on the evaluation episodes (seeds 1000-1019) and reports
  * frac_band_steps: fraction of variable-steps with |v| <= MIN_SIGNIFICANT
  * max_run: longest run of consecutive steps any single variable spent inside the band
  * length: mean episode length (steps until a variable leaves the viability bound, max 1000)

Usage: python mrp_v3_check.py  -> docs/results_v3_mrp/analysis/V3_dynamics_check.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mrp_env import MIN_SIGNIFICANT, MRPEnv, make_reward_fn  # noqa: E402
from runner import CONDITIONS  # noqa: E402

SEEDS = list(range(1000, 1020))


def run():
    rows = []
    for dyn in ("legacy", "v3"):
        for c, p in CONDITIONS.items():
            for pol in ("random", "zero"):
                env = MRPEnv(p["n"], p["budget"], p["noise"], make_reward_fn("DR"), dynamics=dyn)
                rng = np.random.default_rng(0)
                band, lengths, max_run = [], [], 0
                for s in SEEDS:
                    env.reset(s)
                    run_len = np.zeros(p["n"], dtype=int)
                    done, L = False, 0
                    while not done:
                        logits = rng.normal(0, 1, p["n"]) if pol == "random" else np.zeros(p["n"])
                        _, _, term, trunc, _ = env.step(logits)
                        inb = np.abs(env.v) <= MIN_SIGNIFICANT
                        band.append(inb.mean())
                        run_len = np.where(inb, run_len + 1, 0)
                        max_run = max(max_run, int(run_len.max()))
                        L += 1
                        done = term or trunc
                    lengths.append(L)
                rows.append(dict(dynamics=dyn, cond=c, policy=pol, frac_band_steps=float(np.mean(band)),
                                 max_run=max_run, length=float(np.mean(lengths))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = os.environ.get("MRP_RES", os.path.join(HERE, "..", "docs", "results_v3_mrp"))
    out = os.path.join(res, "analysis")
    os.makedirs(out, exist_ok=True)
    df = run()
    df.to_csv(os.path.join(out, "V3_dynamics_check.csv"), index=False)
    print(df.pivot_table(index=["cond", "policy"], columns="dynamics",
                         values=["frac_band_steps", "max_run", "length"]).round(3).to_string())
