"""Figure 3 + diversity table: real-model selection shift vs N and candidate
diversity, from the generated arms.
Needs DATA dir with:
  mtx16/unwm_kgw_scored.jsonl   (unwatermarked iid; rewards)
  mtx16/gumbel_scored.jsonl     (softened shared key; rewards+pvalues)
  jcs/ss_scored.jsonl           (strict shared key)
  jcs/sk_scored.jsonl           (strict subkey; union-Bonferroni pvalues)
  jcs/ss_r.jsonl, jcs/sk_r.jsonl (candidate text for diversity)
Output: figures/real_shift_vs_N.pdf + printed diversity table.
"""
import os, json
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("DATA", os.path.expanduser(
    "~/Desktop/iclr_wm_exp/pod_experiments/results"))
matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                            "axes.labelsize": 9, "legend.fontsize": 8})

def load(f):
    return [json.loads(l) for l in open(os.path.join(DATA, f))]

def RZ(path):
    R, Z = [], []
    for d in load(path):
        r = np.array([x if x == x else -9. for x in d["cands_reward"]], float)
        p = np.array([min(max(x, 1e-300), 1.) if x == x else 1.
                      for x in d.get("cands_pvalue", [1.] * len(r))], float)
        R.append(r); Z.append(norm.isf(p))
    return R, Z

ARMS = {"unwm": RZ("mtx16/unwm_kgw_scored.jsonl"),
        "soft": RZ("mtx16/gumbel_scored.jsonl"),
        "ss": RZ("jcs/ss_scored.jsonl"),
        "sk": RZ("jcs/sk_scored.jsonl")}

def curve(R, Z, N, rng, nsel=80):
    rew, zs = [], []
    for r, z in zip(R, Z):
        m = r > -8; r, z = r[m], z[m]
        if len(r) == 0:
            continue
        for _ in range(nsel):
            ix = rng.choice(len(r), min(N, len(r)), replace=False)
            i = ix[int(np.argmax(r[ix]))]
            rew.append(r[i]); zs.append(z[i])
    return np.mean(rew), np.median(zs)

Ns = [1, 2, 4, 8, 16]
CUR = {a: {"r": [], "z": []} for a in ARMS}
for a, (R, Z) in ARMS.items():
    for N in Ns:
        r, z = curve(R, Z, N, np.random.default_rng(0))
        CUR[a]["r"].append(r); CUR[a]["z"].append(z)

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
STY = {"ss": ("#c0392b", "o", "strict shared key"),
       "sk": ("#1a7a3a", "s", "strict subkey (ours)"),
       "soft": ("#7f9fc9", "^", "softened shared key")}
for a in ["ss", "soft", "sk"]:
    c, mk, lab = STY[a]
    gap = [CUR["unwm"]["r"][i] - CUR[a]["r"][i] for i in range(len(Ns))]
    ax[0].plot(Ns, gap, "-", color=c, marker=mk, ms=4, lw=1.8, label=lab)
ax[0].axhline(0, color="k", lw=.7)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(Ns); ax[0].set_xticklabels(Ns)
ax[0].set_xlabel("$N$"); ax[0].set_ylabel("reward gap to unwatermarked BoN")
ax[0].set_title("Selection shift (real model)"); ax[0].grid(alpha=.3); ax[0].legend()
for a in ["ss", "soft", "sk"]:
    c, mk, lab = STY[a]
    ax[1].plot(Ns, CUR[a]["z"], "-", color=c, marker=mk, ms=4, lw=1.8, label=lab)
ax[1].set_xscale("log", base=2); ax[1].set_xticks(Ns); ax[1].set_xticklabels(Ns)
ax[1].set_xlabel("$N$"); ax[1].set_ylabel("median detection z (selected output)")
ax[1].set_title("Watermark strength after selection"); ax[1].grid(alpha=.3)
fig.tight_layout()
fig.savefig("figures/real_shift_vs_N.pdf", bbox_inches="tight")
print("saved figures/real_shift_vs_N.pdf")

# ---- diversity table ----
def ngrams(t, n=3):
    w = t.split()
    return set(tuple(w[i:i + n]) for i in range(len(w) - n + 1))

print("\nDiversity (within-prompt):")
for tag, fn in [("unwm", "mtx16/unwm_kgw_scored.jsonl"),
                ("softened", "mtx16/gumbel_scored.jsonl"),
                ("ss", "jcs/ss_r.jsonl"), ("sk", "jcs/sk_r.jsonl")]:
    uq, jac, pref = [], [], []
    for d in load(fn):
        c = [x for x in d["cands"] if x and x.strip()]
        uq.append(len(set(c)) / max(1, len(c)))
        for i in range(min(len(c), 8)):
            for j in range(i + 1, min(len(c), 8)):
                a, b = ngrams(c[i]), ngrams(c[j])
                if a or b:
                    jac.append(len(a & b) / max(1, len(a | b)))
                wa, wb = c[i].split(), c[j].split()
                k = 0
                while k < min(len(wa), len(wb)) and wa[k] == wb[k]:
                    k += 1
                pref.append(k)
    print(" %-9s unique=%.2f jaccard=%.4f shared-prefix=%.2f words"
          % (tag, np.mean(uq), np.mean(jac), np.mean(pref)))
