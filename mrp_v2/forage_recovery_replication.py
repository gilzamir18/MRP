"""Reanalysis (no training): RMST of the foraging recovery-test retrained models.

The recovery test (docs/results_v2_noadapt/F_term_rec) retrained every
formulation of the foraging study with termination (same seeds and
configuration, weights saved). This script recomputes, from the per-run JSONs,
the RMST at tau = 500 of the stochastic evaluation of those runs and puts it
next to the RMST of the original runs (FORAGE_F_term_seed_level.csv), giving
the values of the replication table of the paper (Table app_repl).

Per episode RMST = min(length, 500); averaged within seed, then over seeds.
Writes docs/results_v2_noadapt/analysis/FORAGE_recovery_rmst_replication.csv.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

TAU = 500
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "docs", "results_v2_noadapt")
REC = os.path.join(BASE, "F_term_rec")
ANA = os.path.join(BASE, "analysis")

rows = []
for path in sorted(glob.glob(os.path.join(REC, "*__term__s*.json"))):
    m = re.match(r"(F\d)__(.+)__term__s(\d+)\.json$", os.path.basename(path))
    if not m:
        continue
    cond, variant, seed = m.group(1), m.group(2), int(m.group(3))
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    lengths = np.array([ep["length"] for ep in d["eval"]], dtype=float)
    rows.append(dict(cond=cond, variant=variant, seed=seed,
                     rmst=np.minimum(lengths, TAU).mean()))
rep = pd.DataFrame(rows)

orig = pd.read_csv(os.path.join(ANA, "FORAGE_F_term_seed_level.csv"))
orig = orig[orig.kind == "eval"]

def summ(df, tag):
    g = df.groupby(["cond", "variant"]).rmst
    return pd.DataFrame({f"rmst_{tag}": g.mean(), f"sd_{tag}": g.std(ddof=1),
                         f"n_{tag}": g.count()})

out = summ(orig, "original").join(summ(rep, "replication"), how="inner").reset_index()
out["diff"] = out.rmst_replication - out.rmst_original
out["abs_diff"] = out["diff"].abs()
out.to_csv(os.path.join(ANA, "FORAGE_recovery_rmst_replication.csv"), index=False)
print(out.to_string(index=False, float_format=lambda z: f"{z:.2f}"))
print(f"cells: {len(out)}; max |diff| = {out.abs_diff.max():.2f} "
      f"({out.loc[out.abs_diff.idxmax(), 'variant']}, {out.loc[out.abs_diff.idxmax(), 'cond']}); "
      f"cells with |diff| < 20: {(out.abs_diff < 20).sum()}")
