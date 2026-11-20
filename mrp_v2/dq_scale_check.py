"""
dq_scale_check.py — scale of the per-step DQ reward under the uniformly random policy (foraging,
with termination).  Source of the numbers quoted in the text (mean, SD, shift / SD ratios).

2000 episodes per condition (seeds 50000-51999) pooled over all steps, terminal step included.
Estimates from only 20 episodes are noisy (F2: 5-95% range of the SD is 0.008-0.011), which is why
earlier quick estimates disagreed (0.011 vs 0.008 in F2).

    python dq_scale_check.py   -> docs/results_v2/analysis/DQ_scale_random_policy.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import OUT  # noqa: E402
from forage_env import ForageEnv, make_reward_fn, policy_random  # noqa: E402
from forage_run import FCONDS  # noqa: E402

N_EP, SEED0 = 2000, 50_000


def dq_steps(cond, seeds, rng_seed=7):
    c = FCONDS[cond]
    env = ForageEnv(len(seeds), c["k"], make_reward_fn("DQ"), terminate=True, W=9, decay_mu=c["decay_mu"])
    for i, s in enumerate(seeds):
        env.reset_one(i, s)
    rng = np.random.default_rng(rng_seed)
    alive = np.ones(env.E, bool)
    out = []
    while alive.any():
        _, term, trunc, info = env.step(policy_random(env, rng))
        dq = (info["dev_prev"] ** 2 - info["dev"] ** 2).mean(1)
        out.append(dq[alive])
        alive &= ~(term | trunc)
    return np.concatenate(out)


def main():
    rows = []
    for cond in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        r = dq_steps(cond, list(range(SEED0, SEED0 + N_EP)))
        sd = r.std()
        rows.append(dict(cond=cond, k=FCONDS[cond]["k"], decay=FCONDS[cond]["decay_mu"], n_steps=len(r),
                         mean=r.mean(), sd=sd, frac_neg=(r < 0).mean(), mean_plus_001=r.mean() + 0.01,
                         ratio_1_over_sd=1.0 / sd, ratio_01_over_sd=0.1 / sd))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "DQ_scale_random_policy.csv"), index=False)
    print(D.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
