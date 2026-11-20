"""Per-step sign of every foraging formulation under a random policy with termination
(same transitions as forage_grid_sign.py). Output: FORAGE_sign_random_policy.csv"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from analyze import OUT
rows = []
from forage_grid_sign import transitions
from forage_env import or_per_variable
from forage_run import FCONDS

for cond in ["F1", "F2", "F3", "F4", "F5", "F6"]:
    if cond not in FCONDS:
        print(cond, "missing"); continue
    d0, d1 = transitions(cond)
    R = {
        "OR": or_per_variable(d0, d1, 0.5, 1.0).mean(1),
        "OR_noZone": or_per_variable(d0, d1, 0.0, 1.0).mean(1),
        "OR_noPen": or_per_variable(d0, d1, 0.5, 0.0).mean(1),
        "DeadZone": -(np.maximum(0, d1 - 0.5) ** 2).mean(1),
        "DR": -(d1 ** 2).mean(1),
        "DQ": (d0 ** 2).mean(1) - (d1 ** 2).mean(1),
    }
    rows += [dict(cond=cond, reward=k, mean=v.mean(), sd=v.std(), p_pos=(v>0).mean(), p_neg=(v<0).mean()) for k, v in R.items()]
    out = [f"{k}: mean {v.mean():+.3f} sd {v.std():.3f} P>0 {(v>0).mean():.2f} P<0 {(v<0).mean():.2f}" for k, v in R.items()]
    print(cond, len(d0), "\n   " + "\n   ".join(out))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "FORAGE_sign_random_policy.csv"), index=False)
