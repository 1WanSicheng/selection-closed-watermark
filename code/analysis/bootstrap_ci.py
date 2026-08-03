"""Paired bootstrap CIs for the headline numbers (Table 1 and the
gain-retained metric). Resamples the 128 prompts with replacement; the same
prompt indices are used for every arm, so differences and ratios are paired.

Needs DATA dir (see suite_table.py):
  mtx16/unwm_r.jsonl            unwatermarked rewards
  mtx16/gumbel_scored.jsonl     softened arm (reward + p-value)
  jcs/ss_scored.jsonl           shared deterministic key
  jcs/sk_scored.jsonl           candidate-separated subkeys (union p)
"""
import os, json
import numpy as np
from scipy.stats import norm

DATA = os.environ.get("DATA", os.path.expanduser(
    "~/Desktop/iclr_wm_exp/pod_experiments/results"))
B = 10000
rng = np.random.default_rng(7)

def load(f):
    return [json.loads(l) for l in open(os.path.join(DATA, f))]

def arm(path):
    """Per prompt: (reward vector over candidates, p-value vector or None)."""
    out = []
    for d in load(path):
        r = np.array([x if x == x else -9. for x in d["cands_reward"]], float)
        m = r > -8
        p = None
        if "cands_pvalue" in d:
            p = np.array([min(max(x, 1e-300), 1.) if x == x else 1.
                          for x in d["cands_pvalue"]], float)[m]
        out.append((r[m], p))
    return out

ARMS = {
    "unwm":     arm("mtx16/unwm_r.jsonl"),
    "softened": arm("mtx16/gumbel_scored.jsonl"),
    "ss":       arm("jcs/ss_scored.jsonl"),
    "sk":       arm("jcs/sk_scored.jsonl"),
}
NP = min(len(v) for v in ARMS.values())

def stats(name, idx):
    A = ARMS[name]
    single = np.mean([A[i][0].mean() for i in idx])
    bon = np.mean([A[i][0].max() for i in idx])
    zs = None
    if A[0][1] is not None:
        zsingle = np.concatenate([norm.isf(A[i][1]) for i in idx])
        zbon = np.array([norm.isf(A[i][1][int(np.argmax(A[i][0]))]) for i in idx])
        zs = (np.median(zsingle), np.median(zbon))
    return single, bon, zs

full = np.arange(NP)
point = {k: stats(k, full) for k in ARMS}

acc = {k: [] for k in ARMS}
diffs = {"sk_bon-unwm_bon": [], "sk_bon-soft_bon": [], "sk_single-unwm_single": [],
         "ss_bon-unwm_bon": [], "soft_bon-unwm_bon": []}
gr = {"ss": [], "softened": [], "sk": []}
for _ in range(B):
    idx = rng.integers(0, NP, NP)
    st = {k: stats(k, idx) for k in ARMS}
    for k in ARMS:
        acc[k].append(st[k])
    diffs["sk_bon-unwm_bon"].append(st["sk"][1] - st["unwm"][1])
    diffs["sk_bon-soft_bon"].append(st["sk"][1] - st["softened"][1])
    diffs["sk_single-unwm_single"].append(st["sk"][0] - st["unwm"][0])
    diffs["ss_bon-unwm_bon"].append(st["ss"][1] - st["unwm"][1])
    diffs["soft_bon-unwm_bon"].append(st["softened"][1] - st["unwm"][1])
    g_un = st["unwm"][1] - st["unwm"][0]
    for k in gr:
        gr[k].append((st[k][1] - st[k][0]) / g_un)

def ci(v, lab, pt=None, pct=False):
    lo, hi = np.percentile(v, [2.5, 97.5])
    f = (lambda x: "%.0f%%" % (100 * x)) if pct else (lambda x: "%.4f" % x)
    print("%-28s %s  [%s, %s]" % (lab, f(pt) if pt is not None else f(np.mean(v)),
                                  f(lo), f(hi)))

print("=== point estimate  [95%% paired-bootstrap CI],  B=%d, prompts=%d ===" % (B, NP))
for k in ARMS:
    ci([a[0] for a in acc[k]], k + " single reward", point[k][0])
    ci([a[1] for a in acc[k]], k + " hard-BoN16 reward", point[k][1])
    if point[k][2] is not None:
        ci([a[2][0] for a in acc[k]], k + " median z (single)", point[k][2][0])
        ci([a[2][1] for a in acc[k]], k + " median z (post-BoN)", point[k][2][1])
print("--- paired differences ---")
for lab, v in diffs.items():
    ci(v, lab)
print("--- gain retained: (BoN-single)/(BoN-single)_unwm ---")
g_un_pt = point["unwm"][1] - point["unwm"][0]
for k in gr:
    ci(gr[k], k, (point[k][1] - point[k][0]) / g_un_pt, pct=True)
