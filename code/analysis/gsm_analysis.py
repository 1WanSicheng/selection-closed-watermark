"""GSM8K reasoning-scaling analysis.
Metrics per arm: pass@1 (mean over candidates), pass@16 (any correct),
maj@16 (self-consistency vote), maj@k curve, extraction failure rate.
Detection: median z (all candidates), z of the majority-vote winner,
TPR@1% at the domain-matched empirical null threshold, corr(correct, z).
Paired bootstrap CIs over problems for the headline differences.
"""
import json, os, re, sys
from collections import Counter
import numpy as np
from scipy.stats import norm

DATA = os.path.expanduser("~/Desktop/iclr_wm_exp/pod_experiments/results/gsm")

def load(f):
    p = os.path.join(DATA, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None

def norm_num(s):
    s = s.replace(",", "").replace("$", "").rstrip(".").strip()
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else ("%g" % v)
    except ValueError:
        return None

def extract(text):
    m = re.findall(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m: return norm_num(m[0])
    m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    return norm_num(m[-1]) if m else None

def arm_stats(gen, scored=None, nullq=None):
    P1, P16, MAJ, fail = [], [], [], []
    majk = {k: [] for k in range(1, 17)}
    zsel, zall, hitsel, corr = [], [], [], []
    rng = np.random.default_rng(5)
    for li, d in enumerate(gen):
        gold = norm_num(d["gold"])
        ans = [extract(c) for c in d["cands"]]
        fail += [a is None for a in ans]
        ok = np.array([a == gold for a in ans], dtype=float)
        P1.append(ok.mean()); P16.append(float(ok.max()))
        votes = [a for a in ans if a is not None]
        majv = Counter(votes).most_common(1)[0][0] if votes else None
        MAJ.append(float(majv == gold))
        for k in majk:
            accs = []
            for _ in range(40):
                idx = rng.choice(16, k, replace=False)
                v = [ans[i] for i in idx if ans[i] is not None]
                accs.append(float((Counter(v).most_common(1)[0][0] if v else None) == gold))
            majk[k].append(np.mean(accs))
        if scored is not None:
            p = np.array([min(max(x, 1e-300), 1 - 1e-12) if x == x else 1 - 1e-12
                          for x in scored[li]["cands_pvalue"]])
            z = norm.isf(p)
            zall += list(z)
            win = next((i for i, a in enumerate(ans) if a == majv), None)
            if win is not None:
                zsel.append(z[win])
                if nullq is not None: hitsel.append(z[win] > nullq)
            if ok.std() > 0 and z.std() > 0:
                corr.append(np.corrcoef(ok, z)[0, 1])
    out = dict(p1=np.mean(P1), p16=np.mean(P16), maj=np.mean(MAJ),
               fail=np.mean(fail), majk={k: np.mean(v) for k, v in majk.items()},
               P1=P1, P16=P16, MAJ=MAJ)
    if scored is not None:
        out.update(zall=np.median(zall), zsel=np.median(zsel),
                   tprsel=np.mean(hitsel) if hitsel else None,
                   corr=np.mean(corr) if corr else None)
    return out

def nullq_of(f):
    U = load(f)
    if U is None: return None
    z = [norm.isf(min(max(x, 1e-300), 1 - 1e-12)) for d in U for x in d["cands_pvalue"] if x == x]
    return float(np.quantile(z, 0.99))

if __name__ == "__main__":
    qgu, qg, qk = nullq_of("null_gu.jsonl"), nullq_of("null_g.jsonl"), nullq_of("null_kgw.jsonl")
    arms = [("unwm", "unwm.jsonl", None, None),
            ("KGW d2", "kgw.jsonl", "kgw_scored.jsonl", qk),
            ("Gumbel soft", "gsoft.jsonl", "gsoft_scored.jsonl", qg),
            ("Gumbel shared", "gss.jsonl", "gss_scored.jsonl", qg),
            ("Gumbel subkey", "gsk.jsonl", "gsk_scored.jsonl", qgu)]
    res = {}
    print("%-14s %7s %8s %8s %6s | %7s %7s %8s %7s" %
          ("arm", "pass@1", "pass@16", "maj@16", "fail", "z(all)", "z(maj)", "TPR.01", "corr"))
    for name, gf, sf, q in arms:
        gen, sc = load(gf), load(sf) if sf else None
        if gen is None: print(name, "MISSING"); continue
        r = arm_stats(gen, sc, q); res[name] = r
        def fm(v, f):
            return ("%" + f) % v if v is not None else "    ---"
        det = (fm(r["zall"], "7.2f") + fm(r["zsel"], "7.2f") +
               fm(r["tprsel"], "8.3f") + fm(r["corr"], "7.3f")) if sc else "    ---     ---      ---     ---"
        print("%-14s %7.3f %8.3f %8.3f %6.3f | %s" % (name, r["p1"], r["p16"], r["maj"], r["fail"], det))
    # paired bootstrap for headline differences
    rng = np.random.default_rng(9)
    def ci(a, b, key):
        A, B = np.array(res[a][key]), np.array(res[b][key])
        n = min(len(A), len(B)); d = []
        for _ in range(10000):
            idx = rng.integers(0, n, n); d.append(np.mean(A[idx] - B[idx]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("  %s - %s  [%s]: %+.3f  [%+.3f, %+.3f]" % (a, b, key, np.mean(d), lo, hi))
    print("--- paired differences ---")
    for key in ["P1", "MAJ", "P16"]:
        for a in ["Gumbel subkey", "Gumbel soft", "KGW d2", "Gumbel shared"]:
            if a in res: ci(a, "unwm", key)
    print("--- maj@k curve (unwm / shared / subkey / soft / kgw) ---")
    for k in [1, 2, 4, 8, 16]:
        row = [("%.3f" % res[a]["majk"][k]) if a in res else "--" for a in
               ["unwm", "Gumbel shared", "Gumbel subkey", "Gumbel soft", "KGW d2"]]
        print("  k=%2d  " % k + "  ".join(row))
