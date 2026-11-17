"""S6 (docs/prereg_2026-11-16.md, adendo): OR vs DQ without termination, 3M vs 10M; primary = viable fraction."""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_nt import paired_test, holm
RES = os.environ.get("MRP_RES", os.path.join(os.path.dirname(HERE), "docs", "results_v2"))

def load(study, tag):
    rows = []
    for p in glob.glob(os.path.join(RES, study, "*.json")):
        r = json.load(open(p, encoding="utf-8")); j = r["job"]
        if j["tag"] != tag or j["variant"] not in ("OR_fixed", "DQ") or j["cond"] not in ("F3", "F4", "F5", "F6"):
            continue
        for kind in ("eval", "eval_det"):
            ev = r[kind]
            rows.append(dict(steps=j.get("timesteps", 3_000_000), kind=kind, cond=j["cond"], variant=j["variant"], seed=j["seed"],
                             viable=np.mean([e["frac_viable"] for e in ev]), in_zone=np.mean([e["frac_in_zone"] for e in ev]),
                             over=np.mean([e["over_share"] for e in ev]), first_viol=np.mean([min(e["first_viol"], 500) for e in ev])))
    return pd.DataFrame(rows)

D = pd.concat([load("F_noterm", "noterm"), load("F_ksweep", "noterm"), load("F_long_nt", "noterm10M")])
out = []
for (st, kind), g in D.groupby(["steps", "kind"]):
    rows = []
    for c in ("F3", "F4", "F5", "F6"):
        a = g[(g.cond == c) & (g.variant == "OR_fixed")].set_index("seed").viable
        b = g[(g.cond == c) & (g.variant == "DQ")].set_index("seed").viable
        com = a.index.intersection(b.index)
        t = paired_test(a.loc[com].values, b.loc[com].values)
        rows.append(dict(steps=st, kind=kind, cond=c, OR=a.loc[com].mean(), DQ=b.loc[com].mean(), wins_OR=int((a.loc[com] > b.loc[com]).sum()), **t))
    df = pd.DataFrame(rows); df["p_holm"] = holm(df.p_raw.values); out.append(df)
R = pd.concat(out)
R.to_csv(os.path.join(RES, "analysis", "CLARIFY_S6_noterm_contrasts.csv"), index=False)
print(R[["steps", "kind", "cond", "OR", "DQ", "diff", "wins_OR", "n", "p_raw", "p_holm"]].round(3).to_markdown(index=False))
print(D[D.kind == "eval"].groupby(["steps", "cond", "variant"])[["viable", "in_zone", "over", "first_viol"]].mean().round(3).to_markdown())
