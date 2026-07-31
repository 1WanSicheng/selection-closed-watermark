"""Figure 2: union vs aggregate detection under N subkeys, plus the exact
closed form Delta_agg = ln(1+(e^Delta-1)/N) (Prop. 3 / Eq. 1).
Simulation: V-token vocab, per-position logits with spread s (entropy knob);
selected candidate scored under (i) its own subkey (oracle), (ii) union over N,
(iii) column-max aggregate with exactly calibrated Gumbel(ln N) null.
Output: figures/detector_comparison.pdf
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                            "axes.labelsize": 9, "legend.fontsize": 8})
rng = np.random.default_rng(0)
V, T, M = 1000, 200, 800
EULER = 0.5772156649
SG = math.pi / math.sqrt(6)

def gum(sh):
    return -np.log(-np.log(rng.random(sh)))

def run(N, s):
    zs, zu, za, ents = [], [], [], []
    for _ in range(M):
        l = s * rng.standard_normal(V)
        p = np.exp(l - l.max()); p /= p.sum()
        ents.append(-(p * np.log(p)).sum())
        Gj = gum((T, V))                       # selected candidate's own stream
        x = np.argmax(l[None, :] + Gj, axis=1)
        gj = Gj[np.arange(T), x]
        sd = SG * math.sqrt(T)
        z_own = (gj.sum() - T * EULER) / sd
        zs.append(z_own)
        znull = (gum((max(N - 1, 1), T)).sum(1) - T * EULER) / sd
        zu.append(max([z_own] + list(znull)) if N > 1 else z_own)
        gagg = np.maximum(gj, math.log(N - 1) + gum(T)) if N > 1 else gj
        za.append((gagg.sum() - T * (math.log(N) + EULER)) / sd)
    return np.mean(ents), np.mean(zs), np.mean(zu), np.mean(za)

Ns = [2, 4, 8, 16, 32, 64]
DATA = {}
for s, tag in [(0.5, "high"), (3.0, "low")]:
    DATA[tag] = {"single": [], "union": [], "agg": [], "H": None}
    for N in Ns:
        H, zs, zu, za = run(N, s)
        for k, v in [("single", zs), ("union", zu), ("agg", za)]:
            DATA[tag][k].append(v)
        DATA[tag]["H"] = H
    D = np.mean(DATA[tag]["single"]) * SG / math.sqrt(T)   # per-token Delta
    DATA[tag]["pred"] = [math.sqrt(T) * math.log(1 + (math.exp(D) - 1) / N) / SG
                         for N in Ns]

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1))
for ax, tag, ttl in [(axs[0], "high", "High entropy ($H\\approx6.8$ nats)"),
                     (axs[1], "low", "Low entropy ($H\\approx3.4$ nats)")]:
    d = DATA[tag]
    ax.plot(Ns, d["single"], "--", color="#666666", lw=1.6, marker="o", ms=3.5,
            label="oracle (correct subkey)")
    ax.plot(Ns, d["union"], "-", color="#1a7a3a", lw=2, marker="s", ms=4,
            label="union detector")
    ax.plot(Ns, d["agg"], "-", color="#c0392b", lw=2, marker="^", ms=4,
            label="aggregate (column-max)")
    ax.plot(Ns, d["pred"], ":", color="#c0392b", lw=1.4, label="aggregate closed form")
    ax.plot(Ns, [math.sqrt(2 * math.log(N)) for N in Ns], "-.", color="#888888",
            lw=1, label="union threshold $\\sqrt{2\\ln N}$")
    ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("$N$ (candidates / subkeys)"); ax.set_title(ttl); ax.grid(alpha=.3)
axs[0].set_ylabel("detection z-score")
axs[0].legend(loc="center left", framealpha=.9)
fig.tight_layout()
fig.savefig("figures/detector_comparison.pdf", bbox_inches="tight")
print("saved figures/detector_comparison.pdf")
