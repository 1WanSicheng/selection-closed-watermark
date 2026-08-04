"""Selection-head matrix: single / BoN / SBoN(lambda) / BoP(mu) / BoL(b)
over scored candidate pools. Heads are computed exactly where possible:
  SBoN: exact softmax expectation, weights ∝ exp(r/lambda)
  BoP : M ~ Poisson(mu) truncated to >=1, capped at pool size; exact
        order-statistics mixture for the max of a random M-subset
  BoL : argmax(r_i + Laplace(0,b)), Monte Carlo (fixed seed)
Metrics: mean selected reward; median selected z (per-prompt expectation,
median across prompts); mean selected p<0.01 rate where p-values exist.
Usage: python heads_matrix.py <tag:path> [<tag:path> ...]
"""
import json, os, sys
import numpy as np
from scipy.stats import norm, poisson
from math import comb

DATA = os.path.expanduser("~/Desktop/iclr_wm_exp/pod_experiments/results")

def load_arm(path):
    out = []
    for l in open(os.path.join(DATA, path)):
        d = json.loads(l)
        r = np.array([x if x == x else -9. for x in d["cands_reward"]], float)
        m = r > -8
        p = None
        if "cands_pvalue" in d:
            p = np.array([min(max(x, 1e-300), 1. - 1e-12) if x == x else 1. - 1e-12
                          for x in d["cands_pvalue"]], float)[m]
        out.append((r[m], p))
    return out

def head_weights(r, head, par, rng):
    """Return selection-probability vector over the pool for this prompt."""
    n = len(r)
    if head == "single":
        return np.full(n, 1. / n)
    if head == "bon":
        w = np.zeros(n); w[np.argmax(r)] = 1.; return w
    if head == "sbon":
        e = np.exp((r - r.max()) / par); return e / e.sum()
    if head == "bop":
        # Algorithm 3 of arXiv:2506.19248: M = 1 + Poisson(par), hard max;
        # M capped at pool size n (mass of M>=n goes to max of all n)
        order = np.argsort(r)               # ascending
        w = np.zeros(n)
        for m in range(1, n):
            pr_m = poisson.pmf(m - 1, par)
            for j in range(m, n + 1):       # rank j (1-indexed) is subset max
                w[order[j - 1]] += pr_m * comb(j - 1, m - 1) / comb(n, m)
        w[order[n - 1]] += 1. - poisson.cdf(n - 2, par)   # M >= n
        return w / w.sum()
    if head == "bol":
        idx = np.argmax(r[None, :] + rng.laplace(0., par, size=(4000, len(r))), axis=1)
        return np.bincount(idx, minlength=n) / 4000.
    raise ValueError(head)

HEADS = [("single", None), ("bon", None),
         ("sbon", 0.02), ("sbon", 0.1),
         ("bop", 3.), ("bop", 15.),
         ("bol", 0.02), ("bol", 0.1)]

def run(tag, path):
    A = load_arm(path)
    rng = np.random.default_rng(3)
    row = {}
    for head, par in HEADS:
        rs, zs, hits = [], [], []
        for r, p in A:
            w = head_weights(r, head, par, rng)
            rs.append(float(w @ r))
            if p is not None:
                z = norm.isf(p)
                zs.append(float(w @ z))
                hits.append(float(w @ (p < 0.01)))
        key = head + ("" if par is None else str(par))
        row[key] = (np.mean(rs),
                    (np.median(zs) if zs else None),
                    (np.mean(hits) if hits else None))
    return row

if __name__ == "__main__":
    arms = [a.split(":", 1) for a in sys.argv[1:]]
    hdr = ["single", "bon", "sbon0.02", "sbon0.1", "bop3.0", "bop15.0", "bol0.02", "bol0.1"]
    print("%-12s %-8s" % ("arm", "metric") + "".join("%9s" % h for h in hdr))
    for tag, path in arms:
        row = run(tag, path)
        print("%-12s %-8s" % (tag, "reward") + "".join("%9.4f" % row[h][0] for h in hdr))
        if row["bon"][1] is not None:
            print("%-12s %-8s" % ("", "med z") + "".join("%9.2f" % row[h][1] for h in hdr))
            print("%-12s %-8s" % ("", "p<.01") + "".join("%9.3f" % row[h][2] for h in hdr))
