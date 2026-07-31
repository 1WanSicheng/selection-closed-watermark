"""Table 2 (full metric suite) + matched-FPR table (Appendix B).
Needs DATA dir with:
  mtx16/gumbel_scored.jsonl, mtx16/gumbel_ppl.jsonl        (softened arm)
  mtx16/unwm_gumbel_scored.jsonl                            (base-key null)
  mtx16/unwm_ppl.jsonl                                      (unwm PPL ref)
  jcs/ss_scored.jsonl, jcs/ss_ppl.jsonl                     (strict shared)
  jcs/sk_scored.jsonl, jcs/sk_ppl.jsonl                     (strict subkey, union p)
  jcs/unwm_union_full_scored.jsonl                          (union null, 128 prompts)
"""
import os, json
import numpy as np
from scipy.stats import norm
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    TOK = lambda t: max(1, len(_enc.encode(t)))
except Exception:
    TOK = lambda t: max(1, round(len(t.split()) * 1.33))

DATA = os.environ.get("DATA", os.path.expanduser(
    "~/Desktop/iclr_wm_exp/pod_experiments/results"))

def load(f):
    return [json.loads(l) for l in open(os.path.join(DATA, f))]

def arm(path, pplpath=None):
    L = load(path)
    P = load(pplpath) if pplpath else None
    out = []
    for i, d in enumerate(L):
        r = np.array([x if x == x else -9. for x in d["cands_reward"]], float)
        p = np.array([min(max(x, 1e-300), 1.) if x == x else 1.
                      for x in d["cands_pvalue"]], float)
        t = (np.array([TOK(c) for c in d["cands"]], float)
             if "cands" in d else np.full(len(r), 195.))
        q = np.array([x if x == x else np.nan for x in
                      (P[i]["cands_ppl_orig"] if P else [np.nan] * len(r))], float)
        out.append((r, p, t, q))
    return out

ARMS = {
    "softened": (arm("mtx16/gumbel_scored.jsonl", "mtx16/gumbel_ppl.jsonl"),
                 arm("mtx16/unwm_gumbel_scored.jsonl")),
    "ss": (arm("jcs/ss_scored.jsonl", "jcs/ss_ppl.jsonl"),
           arm("mtx16/unwm_gumbel_scored.jsonl")),
    "sk": (arm("jcs/sk_scored.jsonl", "jcs/sk_ppl.jsonl"),
           arm("jcs/unwm_union_full_scored.jsonl")),
}

def collect(A, mode, N=16, nsel=60, seed=1):
    rng = np.random.default_rng(seed); res = []
    for r, p, t, q in A:
        m = r > -8
        r2, p2, t2, q2 = r[m], p[m], t[m], q[m]
        if len(r2) == 0:
            continue
        if mode == "single":
            for i in range(len(r2)):
                res.append((r2[i], p2[i], t2[i], q2[i]))
        else:
            for _ in range(nsel):
                ix = rng.choice(len(r2), min(N, len(r2)), 0)
                i = ix[int(np.argmax(r2[ix]))]
                res.append((r2[i], p2[i], t2[i], q2[i]))
    return np.array(res)

def auroc(pos, neg):
    r = np.concatenate([pos, neg]); o = r.argsort()
    rk = np.empty(len(r)); rk[o] = np.arange(1, len(r) + 1)
    n1 = len(pos)
    return (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))

print("=== Table 2: full suite (nominal thresholds) ===")
print("arm|mode| rew | med z | TPR.05 TPR.01 | FPR.05 FPR.01 | AUROC | ANLPPT | PPL")
for name, (A, NEG) in ARMS.items():
    for mode in ["single", "BoN16"]:
        a = collect(A, mode); n = collect(NEG, mode, seed=2)
        z = norm.isf(a[:, 1])
        print("%-8s|%-6s| %.4f | %5.2f | %.3f %.3f | %.3f %.3f | %.4f | %.4f | %.1f" % (
            name, mode, a[:, 0].mean(), np.median(z),
            np.mean(a[:, 1] < 0.05), np.mean(a[:, 1] < 0.01),
            np.mean(n[:, 1] < 0.05), np.mean(n[:, 1] < 0.01),
            auroc(norm.isf(a[:, 1]), norm.isf(n[:, 1])),
            np.mean(-np.log(a[:, 1]) / a[:, 2]), np.nanmedian(a[:, 3])))

print("\n=== Matched empirical-FPR (threshold = quantile of own null) ===")
for name in ["softened", "sk"]:
    A, NEG = ARMS[name]
    for mode in ["single", "BoN16"]:
        pos = norm.isf(collect(A, mode)[:, 1])
        neg = norm.isf(collect(NEG, mode, seed=2)[:, 1])
        row = [np.mean(pos > np.quantile(neg, 1 - f)) for f in [0.05, 0.01, 0.001]]
        print("%-8s|%-6s| TPR@.05=%.3f TPR@.01=%.3f TPR@.001=%.3f"
              % (name, mode, *row))
