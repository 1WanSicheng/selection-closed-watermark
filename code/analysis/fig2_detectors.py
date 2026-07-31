"""Figure 2: union detection under N subkeys — the union detector matches the
correct-subkey oracle at every N; its only cost is the sqrt(2 ln N) threshold.
Simulation: V-token vocab, per-position logits with spread s (entropy knob).
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
    zs, zu, ents = [], [], []
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
    return np.mean(ents), np.mean(zs), np.mean(zu)

Ns = [2, 4, 8, 16, 32, 64]
DATA = {}
for s, tag in [(0.5, "high"), (3.0, "low")]:
    DATA[tag] = {"single": [], "union": [], "H": None}
    for N in Ns:
        H, zs, zu = run(N, s)
        for k, v in [("single", zs), ("union", zu)]:
            DATA[tag][k].append(v)
        DATA[tag]["H"] = H

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1))
for ax, tag, ttl in [(axs[0], "high", "High entropy ($H\\approx6.8$ nats)"),
                     (axs[1], "low", "Low entropy ($H\\approx3.4$ nats)")]:
    d = DATA[tag]
    ax.plot(Ns, d["single"], "--", color="#666666", lw=1.6, marker="o", ms=3.5,
            label="oracle (correct subkey)")
    ax.plot(Ns, d["union"], "-", color="#1a7a3a", lw=2, marker="s", ms=4,
            label="union detector")
    ax.plot(Ns, [math.sqrt(2 * math.log(N)) for N in Ns], "-.", color="#888888",
            lw=1, label="union threshold $\\sqrt{2\\ln N}$")
    ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("$N$ (candidates / subkeys)"); ax.set_title(ttl); ax.grid(alpha=.3)
axs[0].set_ylabel("detection z-score")
axs[0].legend(loc="center left", framealpha=.9)
fig.tight_layout()
fig.savefig("figures/detector_comparison.pdf", bbox_inches="tight")
print("saved figures/detector_comparison.pdf")
