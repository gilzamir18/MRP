"""
forage_freeze_test.py — does the drift of the adaptive OR parameters need any change in behaviour?

Adaptive OR is trained with the actor FROZEN at its initialisation (near-uniform random policy):
only the critic and the reward parameters (mu_sz, log sigma_sz, mu_beta, log sigma_beta) are
updated by the same PPO objective as in the main study.  If (z, beta) still drift towards
leniency (z up, beta down), the drift is produced by the reward parameters acting on their own
reward (the circularity), not by environment-specific calibration of behaviour.

    python forage_freeze_test.py   -> docs/results_v2/F_freeze/*.json
"""
import json
import os
import sys
import time
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "docs", "results_v2", "F_freeze")


def job(args):
    cond, term, seed = args
    import torch
    torch.set_num_threads(1)
    from forage_env import ForageEnv, make_reward_fn
    from forage_ppo import Agent, train
    from forage_run import FCONDS
    path = os.path.join(OUT, f"{cond}__{'term' if term else 'noterm'}__s{seed}.json")
    if os.path.exists(path):
        return path
    c = FCONDS[cond]
    env = ForageEnv(16, c["k"], make_reward_fn("OR_adapt"), terminate=term, W=9, decay_mu=c["decay_mu"])
    ag = Agent(env.obs_dim, env.n_actions, learn_or=True)
    t0 = time.time()
    log = train(env, ag, 3_000_000, seed=seed, freeze_policy=True)
    res = {"cond": cond, "terminate": term, "seed": seed, "or_traj": log["or_traj"],
           "or_final": ag.or_summary(), "mean_ep_len": float(sum(log["ep_len"]) / max(1, len(log["ep_len"]))),
           "wall_s": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return path


if __name__ == "__main__":
    jobs = [(c, t, s) for c in ["F1", "F2", "F3", "F4"] for t in (True, False) for s in (42, 43, 44)]
    with get_context("spawn").Pool(4) as pool:
        for p in pool.imap_unordered(job, jobs):
            print("done", os.path.basename(p), flush=True)
