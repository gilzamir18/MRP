"""
r2_dq_return_compare.py — descriptive follow-up to R2-C (NOT pre-registered; 2026-11-21).
On the same evaluation episodes, compares the discounted DQ return (gamma = 0.99) obtained by the
behaviour of DQ-trained agents with that obtained by the behaviour of OR-trained agents (both with
termination, R2_term_diag, same training seeds).  If DQ agents that die early obtain at least the
DQ return of OR agents that survive longer, DQ does not reward the extra survival.
Output: docs/results_v2/analysis/R2C_dq_return_compare.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "docs", "results_v2")

rows = []
for f in glob.glob(os.path.join(RES, "R2_term_diag", "*.json")):
    r = json.load(open(f))
    j = r["job"]
    for pol in ("stoch", "det"):
        d = r["diag"][pol]
        rows.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"], policy=pol,
                         ret_dq=np.mean([e["ret_dq"] for e in d]), length=np.mean([e["length"] for e in d])))
df = pd.DataFrame(rows)
out = []
for pol in ("stoch", "det"):
    for c in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        sub = df[(df.policy == pol) & (df.cond == c)]
        a = sub[sub.variant == "DQ"].set_index("seed")
        b = sub[sub.variant == "OR_fixed"].set_index("seed")
        s = a.index.intersection(b.index)
        d = a.ret_dq[s] - b.ret_dq[s]
        out.append(dict(policy=pol, cond=c, n=len(s), len_DQ=a.length[s].mean(), len_OR=b.length[s].mean(),
                        ret_DQ=a.ret_dq[s].mean(), ret_OR=b.ret_dq[s].mean(), diff=d.mean(),
                        n_DQ_higher=int((d > 0).sum()), p=wilcoxon(d, method="exact").pvalue))
o = pd.DataFrame(out)
o.to_csv(os.path.join(RES, "analysis", "R2C_dq_return_compare.csv"), index=False)
print(o.round(4).to_string(index=False))
