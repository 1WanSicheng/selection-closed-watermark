"""Figure 1: toy selection-shift. Marginal-exact candidates with tunable
correlation rho (Gaussian copula); Best-of-N by noisy reward; TV distance of the
selected-output distribution vs unwatermarked BoN (rho=0).
Output: figures/toy_select_shift.pdf
"""
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                            "axes.labelsize": 9, "legend.fontsize": 8})
rng = np.random.default_rng(0)
K = 5                                   # coarse answer space
R = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # true reward per answer
SIG = 0.20                               # reward-model noise

def seldist(N, rho, M=250000):
    if N == 1:
        A = (np.floor(norm.cdf(rng.standard_normal(M)) * K)).astype(int).clip(0, K - 1)
        return np.bincount(A, minlength=K) / M
    cov = (1 - rho) * np.eye(N) + rho * np.ones((N, N))
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((M, N)) @ L.T            # correlated gaussians
    A = (np.floor(norm.cdf(Z) * K)).astype(int).clip(0, K - 1)  # exact marginals
    rew = R[A] + rng.normal(0, SIG, (M, N))
    sel = A[np.arange(M), np.argmax(rew, 1)]         # Best-of-N by reward
    return np.bincount(sel, minlength=K) / M

def tv(a, b):
    return 0.5 * np.abs(a - b).sum()

Ns = [1, 2, 4, 8, 16, 32, 64]
base = {N: seldist(N, 0.0) for N in Ns}
plt.figure(figsize=(4.6, 3.2))
for r, c, mk, lab in [(0.9, "#c0392b", "o", "shared key, $\\rho=0.9$"),
                      (0.6, "#e67e22", "^", "$\\rho=0.6$"),
                      (0.3, "#8e7cc3", "v", "$\\rho=0.3$"),
                      (0.0, "#1a7a3a", "s", "subkeys (independent, $\\rho=0$)")]:
    y = [tv(seldist(N, r), base[N]) for N in Ns]
    plt.plot(Ns, y, "-", color=c, marker=mk, ms=4, lw=1.8, label=lab)
plt.xscale("log", base=2); plt.xticks(Ns, Ns)
plt.xlabel("$N$ (Best-of-$N$ candidates)")
plt.ylabel("TV$(\\,$selected output, unwatermarked BoN$\\,)$")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig("figures/toy_select_shift.pdf", bbox_inches="tight")
print("saved figures/toy_select_shift.pdf")
