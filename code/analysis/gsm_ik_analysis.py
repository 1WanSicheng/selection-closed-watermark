"""Index-known vs index-blind (union) detection on GSM8K short answers.
Per-key scores from bounded_detect (slog per key, text re-encoded — the
deployed verifier path). Correct key of text position ci = perm[ci]
(decode-matched, avoids the beam-sort trap).
- index-known: z at the correct key; threshold = single-key null 99%
  (unwm texts, key-0 column).
- union: max_k z with Bonferroni-style empirical null (max over 16 keys).
Reported on all candidates (single) and on the majority-vote winner.
"""
import json, os, re, sys
from collections import Counter
import numpy as np

DATA = os.path.expanduser("~/Desktop/iclr_wm_exp/pod_experiments/results/gsm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_analysis import extract, norm_num

def load(f):
    return [json.loads(l) for l in open(os.path.join(DATA, f))]

def zlog(c, k):
    n = c["n"]
    return (c["keys"][k]["slog"] - n) / np.sqrt(n)

G = load("gsk_ids.jsonl")          # texts + gold
S = load("gsk_perkey.jsonl")       # per-key scores (order matches gsk_ids cands)
U = load("unwm_perkey.jsonl")      # null per-key scores
P = [json.loads(l) for l in open(os.path.join(DATA, "gsk_perm.json"))]

null_single = np.array([zlog(c, 0) for d in U for c in d["sc"] if c])
null_union = np.array([max(zlog(c, k) for k in range(16)) for d in U for c in d["sc"] if c])
thr_s, thr_u = np.quantile(null_single, 0.99), np.quantile(null_union, 0.99)

ik_s, ik_m, un_s, un_m = [], [], [], []
hit_ik_s, hit_ik_m, hit_un_s, hit_un_m = [], [], [], []
permfail = 0
for dg, ds_, perm in zip(G, S, P):
    gold = norm_num(dg["gold"])
    ans = [extract(c) for c in dg["cands"]]
    votes = [a for a in ans if a is not None]
    majv = Counter(votes).most_common(1)[0][0] if votes else None
    win = next((i for i, a in enumerate(ans) if a == majv), None)
    for ci, c in enumerate(ds_["sc"]):
        if not c: continue
        k = perm[ci]
        if k < 0: permfail += 1; continue
        zik = zlog(c, k)
        zun = max(zlog(c, kk) for kk in range(16))
        ik_s.append(zik); un_s.append(zun)
        hit_ik_s.append(zik > thr_s); hit_un_s.append(zun > thr_u)
        if ci == win:
            ik_m.append(zik); un_m.append(zun)
            hit_ik_m.append(zik > thr_s); hit_un_m.append(zun > thr_u)
print("perm failures: %d   null thresholds: single %.2f  union %.2f" % (permfail, thr_s, thr_u))
print("%-22s %9s %9s %10s %10s" % ("detector", "z_single", "z_maj", "TPR.01_s", "TPR.01_maj"))
print("%-22s %9.2f %9.2f %10.3f %10.3f" % ("index-known (1 key)",
      np.median(ik_s), np.median(ik_m), np.mean(hit_ik_s), np.mean(hit_ik_m)))
print("%-22s %9.2f %9.2f %10.3f %10.3f" % ("union (16 keys)",
      np.median(un_s), np.median(un_m), np.mean(hit_un_s), np.mean(hit_un_m)))
