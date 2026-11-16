"""
reanalyze_noadapt.py — re-run the MRP and foraging analyses with Adaptive OR and Outer-loop OR removed.

The paper no longer contains the parameter-adaptation variants, so the Holm families and Friedman
tests must be recomputed without them.  Run with MRP_RES pointing at a copy of the results that
contains no *__OR_adapt__* / *__OR_outer__* files (docs/results_v2_noadapt, built with hard links):

    MRP_RES=../docs/results_v2_noadapt python reanalyze_noadapt.py

Outputs go to $MRP_RES/analysis; the original docs/results_v2/analysis is not touched.
"""
import os
import sys

assert "noadapt" in os.environ.get("MRP_RES", ""), "set MRP_RES to the no-adapt results copy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import experiments  # noqa: E402

for lst in (experiments.MAIN, experiments.SIGN, experiments.GEOM, experiments.ALL_VARIANTS):
    while "OR_adapt" in lst:
        lst.remove("OR_adapt")

import analyze  # noqa: E402

analyze.analyze_params = lambda *a, **k: None   # section about the learned parameters only


def _strip(mod):
    mod.VARIANTS = [v for v in mod.VARIANTS if v != "OR_adapt"]
    mod.F1 = [p for p in mod.F1 if "OR_adapt" not in p]
    mod.F2 = [p for p in mod.F2 if "OR_adapt" not in p]


if __name__ == "__main__":
    which = sys.argv[1:] or ["mrp", "nt", "forage"]
    if "mrp" in which:
        analyze.main()
    if "nt" in which:
        import analyze_nt
        _strip(analyze_nt)
        analyze_nt.main()
    if "forage" in which:
        import analyze_forage
        _strip(analyze_forage)
        analyze_forage.posthoc_and_params.__globals__["STUDIES"] = analyze_forage.STUDIES
        analyze_forage.main()
