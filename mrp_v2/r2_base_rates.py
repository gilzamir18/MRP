"""R2-F (prereg/prereg_2026-11-20.md): base rates of the F6 grid predictions P1-P4
(prereg_2026-11-16): probability that each prediction holds if its cells were chosen at random
among the 25 cells (hypergeometric, all chosen cells must satisfy the criterion)."""
import os
from math import comb
import pandas as pd
A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "results_v2", "analysis")
g = pd.read_csv(os.path.join(A, "CLARIFY_S4_F6grid.csv"))
RANDOM, ALIVE = 67.4, 68.0
best = g.loc[g["mean"].idxmax()]
crit = {
    "P1 below random (5 cells)": (g["mean"] < RANDOM, 5),
    "P2 top-5 have |mean r| <= 0.5": ((g.mean_r >= -0.5) & (g.mean_r <= 0.5), 5),
    "P3 default within 1 SD of best": (g["mean"] >= best["mean"] - best["std"], 1),
    "P4 within +-25% of Alive (5 cells)": ((g["mean"] - ALIVE).abs() <= 0.25 * ALIVE, 5),
}
rows = []
for name, (mask, m) in crit.items():
    n = int(mask.sum())
    rows.append(dict(prediction=name, cells_satisfying=n, of=len(g), cells_in_prediction=m,
                     base_rate=comb(n, m) / comb(len(g), m)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(A, "R2F_grid_base_rates.csv"), index=False)
print(df.to_string(index=False))
print("joint (P1..P4 independent picks):", df.base_rate.prod())
