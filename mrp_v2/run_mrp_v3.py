"""
run_mrp_v3.py — single entry point for the MRP re-run with the corrected (v3) dynamics.

Why: the legacy dynamics have an absorbing state (a variable with |v| <= 1e-3 never changes
again; see mrp_env.py).  v3 removes it.  Everything here runs with MRP_DYNAMICS=v3 and writes
to docs/results_v3_mrp; docs/results_v2* (legacy dynamics) are never touched.

Adaptive OR is not run (removed from the paper).  Hyperparameters are the defaults of
runner.py (the paper's learning agents were never tuned per formulation).

Usage (from mrp_v2/):
    python run_mrp_v3.py pilot              # 1 short job -> docs/results_v3_mrp_pilot (smoke test)
    python run_mrp_v3.py run [--workers 10] # all training stages (resumable)
    python run_mrp_v3.py probes             # reference policies, reward sign probe, dynamics check
    python run_mrp_v3.py analyze            # analyze.py + analyze_nt.py on docs/results_v3_mrp
    python run_mrp_v3.py all                # run + probes + analyze

Stages run, in this order (see experiments.py for definitions):
    L, L2, LG, LN   permutation-equivariant actor (main, sign, 3e5 steps, geometry, reward norm.)
    NT, NT2         no-termination control
    A               the original 2x64 MLP (documents that it does not learn the MRP)
    N               observation-noise manipulations (MLP)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES_V3 = os.path.abspath(os.path.join(HERE, "..", "docs", "results_v3_mrp"))
RES_PILOT = os.path.abspath(os.path.join(HERE, "..", "docs", "results_v3_mrp_pilot"))
STAGE_ORDER = ["L", "L2", "LG", "LN", "NT", "NT2", "A", "N"]

cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
os.environ["MRP_DYNAMICS"] = "v3"
os.environ["MRP_RES"] = RES_PILOT if cmd == "pilot" else RES_V3   # read by analyze.py at import
sys.path.insert(0, HERE)

import experiments  # noqa: E402
from mrp_env import MRPEnv, make_reward_fn  # noqa: E402

for lst in (experiments.MAIN, experiments.SIGN, experiments.GEOM, experiments.ALL_VARIANTS,
            experiments.NT_VARIANTS):
    while "OR_adapt" in lst:
        lst.remove("OR_adapt")

assert MRPEnv(2, 1.0, 0.0, make_reward_fn("DR")).dynamics == "v3", "dynamics switch failed"


def workers_arg(default=10):
    if "--workers" in sys.argv:
        return int(sys.argv[sys.argv.index("--workers") + 1])
    return default


def run_stages():
    from runner import run_many
    os.makedirs(RES_V3, exist_ok=True)
    w = workers_arg()
    assert w <= 10, "this machine crashed with 15 workers; use <= 10"
    for st in STAGE_ORDER:
        run_many(experiments.STAGES[st](RES_V3), workers=w, label=st)


def pilot():
    from runner import run_job
    job = dict(study="L_learn", out=RES_PILOT, cond="C1", variant="OR_fixed", seed=42,
               arch="shared", tag="pilot", timesteps=5_000)
    print(run_job(job))
    import json, glob
    p = glob.glob(os.path.join(RES_PILOT, "L_learn", "*.json"))[0]
    r = json.load(open(p))
    print("dynamics in json:", r["dynamics"], "| eval lengths:", [e["length"] for e in r["eval"]][:5])


def probes():
    import runpy
    for script in ("policy_bounds.py", "reward_scale_probe.py", "mrp_v3_check.py"):
        print("==", script, flush=True)
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")


def analyze():
    import analyze as A
    A.analyze_params = lambda *a, **k: None   # Adaptive OR removed from the paper
    A.analyze_tuning = lambda *a, **k: None   # no tuning stage in the re-run
    A.main()
    import analyze_nt as N
    N.VARIANTS = [v for v in N.VARIANTS if v != "OR_adapt"]
    N.F1 = [p for p in N.F1 if "OR_adapt" not in p]
    N.F2 = [p for p in N.F2 if "OR_adapt" not in p]
    N.main()


if __name__ == "__main__":
    if cmd == "pilot":
        pilot()
    elif cmd == "run":
        run_stages()
    elif cmd == "probes":
        probes()
    elif cmd == "analyze":
        analyze()
    elif cmd == "all":
        run_stages()
        probes()
        analyze()
    else:
        raise SystemExit(__doc__)
