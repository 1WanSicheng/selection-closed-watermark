"""Selector-family analysis (Appendix B): closure check + watermark impact for
seven key-free selectors (single, random, hard/soft/Exp-BoN, rejection sampling
with random scan order, min-PPL rerank) and two key-dependent stress selectors
(adversarial min-z evasion, WaterMax-style max-z boosting).
Closure check = reward gap of each arm vs unwatermarked under the SAME selector.
Note: the beam wrapper returns candidates sorted by sequence likelihood, so any
order-sensitive selector must randomize scan order to realize iid semantics
(the likelihood-rank gradient itself is reported separately).
Needs $DATA as in suite_table.py.
"""
import os, json
import numpy as np
from scipy.stats import norm

DATA = os.environ.get("DATA", os.path.expanduser(
    "~/Desktop/iclr_wm_exp/pod_experiments/results"))
LAM, THETA = 0.02, 0.0679   # soft/Exp temperature; rejection threshold (unwm median)

def load(f):
    return [json.loads(l) for l in open(os.path.join(DATA, f))]

def join(scored, pplfile):
    S = load(scored); P = load(pplfile) if pplfile else None; out = []
    for i, d in enumerate(S):
        r = np.array([x if x == x else -9. for x in d["cands_reward"]], float)
        p = np.array([min(max(x, 1e-300), 1 - 1e-12) if x == x else 1 - 1e-12
                      for x in d["cands_pvalue"]], float)
        q = np.array([x if x == x else np.nan for x in
                      (P[i]["cands_ppl_orig"] if P else [np.nan] * len(r))], float)
        out.append((r, norm.isf(p), q))
    return out

ARMS = {"unwm": join("mtx16/unwm_gumbel_scored.jsonl", "mtx16/unwm_ppl.jsonl"),
        "softened": join("mtx16/gumbel_scored.jsonl", "mtx16/gumbel_ppl.jsonl"),
        "ss": join("jcs/ss_scored.jsonl", "jcs/ss_ppl.jsonl"),
        "sk": join("jcs/sk_scored.jsonl", "jcs/sk_ppl.jsonl")}
NULLS = {"softened": ARMS["unwm"], "ss": ARMS["unwm"],
         "sk": join("jcs/unwm_union_full_scored.jsonl", "mtx16/unwm_ppl.jsonl")}

def select(A, rule, rng, draws=200, shuffles=20):
    rew, zs = [], []
    for r, z, q in A:
        m = r > -8; r, z, q = r[m], z[m], q[m]
        if len(r) == 0:
            continue
        if rule == "single":
            rew.append(r.mean()); zs.extend(z.tolist()); continue
        if rule == "random":
            i = rng.integers(len(r)); rew.append(r[i]); zs.append(z[i]); continue
        if rule in ("soft", "exp"):
            E = (rng.gumbel(size=(draws, len(r))) if rule == "soft"
                 else rng.exponential(size=(draws, len(r))))
            idx = np.argmax(r / LAM + E, 1)
            rew.append(r[idx].mean()); zs.append(np.median(z[idx])); continue
        if rule == "reject":  # iid semantics: random scan order
            rr, zz = [], []
            for _ in range(shuffles):
                perm = rng.permutation(len(r))
                ok = perm[np.where(r[perm] >= THETA)[0]]
                i = int(ok[0]) if len(ok) else int(perm[0])
                rr.append(r[i]); zz.append(z[i])
            rew.append(np.mean(rr)); zs.append(np.median(zz)); continue
        i = {"hard": lambda: int(np.argmax(r)),
             "minppl": lambda: int(np.nanargmin(q)) if not np.all(np.isnan(q)) else 0,
             "advminz": lambda: int(np.argmin(z)),
             "wmmaxz": lambda: int(np.argmax(z))}[rule]()
        rew.append(r[i]); zs.append(z[i])
    return np.mean(rew), np.median(zs), np.array(zs)

RULES = ["single", "random", "hard", "soft", "exp", "reject", "minppl",
         "advminz", "wmmaxz"]
KEYFREE = {"single", "random", "hard", "soft", "exp", "reject", "minppl"}
print("=== closure: reward gap (arm - unwm, same selector) ===")
U = {}
for ru in RULES:
    U[ru] = select(ARMS["unwm"], ru, np.random.default_rng(0))
    g = [select(ARMS[a], ru, np.random.default_rng(0))[0] - U[ru][0]
         for a in ["ss", "softened", "sk"]]
    print("%-8s key-free=%-3s unwm=%.4f  ss%+.4f softened%+.4f sk%+.4f"
          % (ru, "yes" if ru in KEYFREE else "NO", U[ru][0], *g))
print("\n=== watermark after selection: median z | TPR@empFPR=.01 ===")
for ru in RULES:
    row = []
    for a in ["ss", "softened", "sk"]:
        _, mz, zs = select(ARMS[a], ru, np.random.default_rng(0))
        _, _, zn = select(NULLS[a], ru, np.random.default_rng(1))
        row.append((mz, np.mean(zs > np.quantile(zn, 0.99))))
    print("%-8s ss %5.2f/%.3f  softened %5.2f/%.3f  sk %5.2f/%.3f"
          % (ru, *row[0], *row[1], *row[2]))
