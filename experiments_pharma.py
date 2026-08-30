# -*- coding: utf-8 -*-
"""
Example 8 (NEW) — Pharmaceutical application:
Dose-response curve modeling in drug screening with GP and the Nyström approximation.
Produces: figures/fig8.png and results_pharma.json
"""
import numpy as np
import json
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import arabic_reshaper
from bidi.algorithm import get_display

from scipy.optimize import curve_fit
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Persian plotting ----------
FONT_PATH = "fonts/Vazirmatn-Regular.ttf"
font_manager.fontManager.addfont(FONT_PATH)
FP = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = FP.get_name()
plt.rcParams["axes.unicode_minus"] = False

def fa(s):
    return get_display(arabic_reshaper.reshape(s))

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

def fad(s):
    return str(s).translate(FA_DIGITS)

C1, C2, C3 = "#1f77b4", "#d62728", "#2ca02c"

def rbf(X, Z, l):
    return np.exp(-cdist(X, Z, "sqeuclidean") / (2.0 * l ** 2))

def hill(x, emax, lec50, h):
    """Hill equation on log10-dose scale: E(d) = Emax / (1 + (EC50/d)^h)"""
    return emax / (1.0 + 10 ** ((lec50 - x) * h))

def ec50_from_gp_mean(gx, mu):
    """Estimate log10(EC50) as the crossing of the posterior mean with 50% of its maximum."""
    thr = 0.5 * mu.max()
    idx = np.where(mu >= thr)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return gx[0]
    t = (thr - mu[i - 1]) / (mu[i] - mu[i - 1])
    return gx[i - 1] + t * (gx[i] - gx[i - 1])

def fit_gp_1d(x, y, n_restarts=8, seed=0):
    xmin, xmax = x.min(), x.max()
    xn = (x - xmin) / (xmax - xmin)
    kern = C(1.0, (0.1, 10.0)) * RBF(1.0, (0.05, 1.0)) + WhiteKernel(0.05, (1e-4, 0.5))
    gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=n_restarts, random_state=seed)
    gp.fit(xn.reshape(-1, 1), y)
    return gp, xmin, xmax

def timeit(fn, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best

results = {}

# ==================================================================
# (a) Single compound: GP vs parametric Hill fit
# ==================================================================
def run_single(lec50, h, seed, n_doses=12, noise=0.05):
    r = np.random.default_rng(seed)
    xd = np.logspace(-9, -4, n_doses)
    x = np.log10(xd)
    yt = hill(x, 1.0, lec50, h)
    y = yt + r.normal(0, noise, n_doses)
    gx = np.linspace(x.min() - 0.3, x.max() + 0.3, 300)
    gt = hill(gx, 1.0, lec50, h)
    # GP
    gp, xmin, xmax = fit_gp_1d(x, y)
    gxn = (gx - xmin) / (xmax - xmin)
    mu, sd = gp.predict(gxn.reshape(-1, 1), return_std=True)
    mae_g = mean_absolute_error(gt, mu)
    mse_g = mean_squared_error(gt, mu)
    ll = -0.5 * (np.log(2 * np.pi * sd ** 2) + ((gt - mu) ** 2) / sd ** 2)
    nll = -float(np.mean(ll))
    ec50_g = ec50_from_gp_mean(gx, mu)
    # Parametric Hill fit
    ok_h = True
    try:
        popt, _ = curve_fit(hill, x, y, p0=[1.0, lec50, 1.0],
                            bounds=([0.5, -9.5, 0.3], [1.5, -3.5, 4.0]))
        yh = hill(gx, *popt)
        lec_h = popt[1]
    except Exception:
        ok_h = False
        yh = np.full_like(gx, np.nan)
        lec_h = np.nan
    out = {
        "lec50_true": lec50, "h": h,
        "gp_mae": round(float(mae_g), 3), "gp_mse": round(float(mse_g), 3),
        "gp_nll": round(nll, 3),
        "hill_mae": round(float(mean_absolute_error(gt, yh)), 3) if ok_h else None,
        "hill_mse": round(float(mean_squared_error(gt, yh)), 3) if ok_h else None,
        "ec50_gp": round(float(ec50_g), 3),
        "ec50_hill": round(float(lec_h), 3) if ok_h else None,
        "hill_ok": ok_h,
    }
    print(f"[Pharma-a] {seed}: lec50={lec50} h={h} | GP: MAE={mae_g:.3f} MSE={mse_g:.3f} NLL={nll:.3f} "
          f"ec50={ec50_g:.3f} | Hill: MAE={out['hill_mae']} ec50={lec_h:.3f}")
    return out, dict(gx=gx, gt=gt, x=x, y=y, mu=mu, sd=sd, yh=yh, lec50=lec50)

resA, visA = run_single(-6.3, 1.2, seed=7)
resB, visB = run_single(-5.1, 0.7, seed=8)
results["single"] = {"A": resA, "B": resB}

# ==================================================================
# (b) EC50 recovery across 30 simulated compounds
# ==================================================================
def run_ec50_bench(n_comp=30, noise=0.05):
    rng = np.random.default_rng(11)
    xd = np.logspace(-9, -4, 12)
    x = np.log10(xd)
    ests = {"true": [], "gp": [], "hill": []}
    for i in range(n_comp):
        lec = rng.uniform(-7.5, -5.0)
        h = rng.uniform(0.7, 2.5)
        r = np.random.default_rng(100 + i)
        yt = hill(x, 1.0, lec, h)
        y = yt + r.normal(0, noise, len(x))
        gp, xmin, xmax = fit_gp_1d(x, y, n_restarts=4, seed=i)
        gx = np.linspace(x.min() - 0.3, x.max() + 0.3, 500)
        gxn = (gx - xmin) / (xmax - xmin)
        mu = gp.predict(gxn.reshape(-1, 1))
        ec_g = ec50_from_gp_mean(gx, mu)
        try:
            popt, _ = curve_fit(hill, x, y, p0=[1.0, lec, 1.0],
                                bounds=([0.5, -9.5, 0.3], [1.5, -3.5, 4.0]))
            ec_h = popt[1]
        except Exception:
            ec_h = np.nan
        ests["true"].append(lec)
        ests["gp"].append(ec_g)
        ests["hill"].append(ec_h)
    g = np.array(ests["gp"]); h_ = np.array(ests["hill"]); t = np.array(ests["true"])
    med_g = float(np.nanmedian(np.abs(g - t)))
    med_h = float(np.nanmedian(np.abs(h_ - t)))
    print(f"[Pharma-b] median |err| log10(EC50): GP={med_g:.3f} Hill={med_h:.3f} "
          f"(n_gp_ok={np.isfinite(g).sum()}, n_hill_ok={np.isfinite(h_).sum()})")
    return {"n": n_comp, "gp_med": round(med_g, 3), "hill_med": round(med_h, 3),
            "true": [round(v, 3) for v in ests["true"]],
            "gp": [round(v, 3) if np.isfinite(v) else None for v in ests["gp"]],
            "hill": [round(v, 3) if np.isfinite(v) else None for v in ests["hill"]]}

results["ec50"] = run_ec50_bench()

# ---------- figure 8 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
ax = axes[0]
v = visA
ax.plot(v["gx"], v["gt"], "-", color="black", lw=1.6, label=fa("منحنی واقعی (مدل هیل)"))
ax.scatter(v["x"], v["y"], s=24, c=C1, edgecolor="k", zorder=5, label=fa("داده‌های سنجش"))
ax.plot(v["gx"], v["mu"], color=C3, lw=1.9, label=fa("GP (میانگین پسین)"))
ax.fill_between(v["gx"], v["mu"] - 2 * v["sd"], v["mu"] + 2 * v["sd"],
                color=C3, alpha=0.18, label=fa("فاصله اطمینان ۹۵٪"))
ax.plot(v["gx"], v["yh"], "--", color=C2, lw=1.6, label=fa("برازش پارامتری هیل"))
ax.axvline(v["lec50"], ls=":", color="gray", lw=1.4, label=fa("EC50 واقعی"))
ax.set_xlabel(fa("log10 دز (مولار)"))
ax.set_ylabel(fa("پاسخ نرمال‌شده"))
ax.set_title(fa(f"(الف) منحنی دز-پاسخ ترکیب A — MAE(GP)={resA['gp_mae']}، MAE(هیل)={resA['hill_mae']}"),
             fontsize=10.5)
ax.legend(fontsize=8)
ax.set_xlim(v["gx"].min(), v["gx"].max())

ax = axes[1]
e = results["ec50"]
t = np.array(e["true"]); g = np.array([v if v is not None else np.nan for v in e["gp"]])
h_ = np.array([v if v is not None else np.nan for v in e["hill"]])
ax.scatter(t, g, s=34, c=C3, marker="o", edgecolor="k", label=fa("برآورد GP"))
ax.scatter(t, h_, s=34, c=C2, marker="s", edgecolor="k", label=fa("برآورد پارامتری هیل"))
lims = [-7.7, -4.8]
ax.plot(lims, lims, "-", color="black", lw=1, alpha=0.7)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(fa("log10(EC50) واقعی"))
ax.set_ylabel(fa("log10(EC50) برآوردشده"))
ax.set_title(fa(f"(ب) بازیابی توان ترکیب — خطای میانه: GP={e['gp_med']}، هیل={e['hill_med']}"),
             fontsize=10.5)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

fig.suptitle(fa("مثال ۸: مدل‌سازی دز-پاسخ در غربالگری دارویی با فرآیند گوسی"), fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig8.png", dpi=200)
plt.close(fig)
print("[fig] fig8.png saved")

# ==================================================================
# (c) High-throughput screening: full GP vs Nyström
# ==================================================================
N_LIST = [300, 1000, 3000]
M = 20
K_LIB = 5000
LELL, SIG = 0.5, 0.05
LEEC, HH = -6.3, 1.2

hts = {"n": N_LIST, "K": K_LIB, "m": M,
       "full_fit": [], "ny_fit": [], "full_mae": [], "ny_mae": []}
for n in N_LIST:
    r = np.random.default_rng(50)
    xd = np.logspace(-9, -4, n)
    x = np.log10(xd)
    xmin, xmax = x.min(), x.max()
    xn = (x - xmin) / (xmax - xmin)
    X = xn.reshape(-1, 1)
    yt = hill(x, 1.0, LEEC, HH)
    y = yt + r.normal(0, SIG, n)
    gx = np.linspace(xmin, xmax, 300)
    gxn = (gx - xmin) / (xmax - xmin)
    gt = hill(gx, 1.0, LEEC, HH)

    def fit_full():
        K = rbf(X, X, LELL) + SIG ** 2 * np.eye(n)
        L = cholesky(K, lower=True)
        return cho_solve((L, True), y)
    tf = timeit(fit_full)
    alpha = fit_full()

    idx = r.choice(n, M, replace=False)
    Xm = X[idx]
    def fit_ny():
        Kmm = rbf(Xm, Xm, LELL) + 1e-9 * np.eye(M)
        Knm = rbf(X, Xm, LELL)
        A = Knm.T @ Knm + SIG ** 2 * Kmm
        return np.linalg.solve(A, Knm.T @ y)
    tn = timeit(fit_ny)
    a_ny = fit_ny()

    k_ = rbf(gxn.reshape(-1, 1), X, LELL)
    mu_f = k_ @ alpha
    km = rbf(gxn.reshape(-1, 1), Xm, LELL)
    mu_n = km @ a_ny

    hts["full_fit"].append(round(tf, 3))
    hts["ny_fit"].append(round(tn, 4))
    hts["full_mae"].append(round(float(mean_absolute_error(gt, mu_f)), 3))
    hts["ny_mae"].append(round(float(mean_absolute_error(gt, mu_n)), 3))
    print(f"[Pharma-c] n={n}: full={tf:.3f}s ny={tn:.4f}s | MAE full={hts['full_mae'][-1]} "
          f"ny={hts['ny_mae'][-1]} | speedup={tf/tn:.0f}x")

# memory (analytic)
n_max = N_LIST[-1]
hts["mem_full_gb"] = round(K_LIB * n_max ** 2 * 8 / 1e9, 1)
hts["mem_ny_gb"] = round(K_LIB * (n_max * M + M ** 2) * 8 / 1e9, 2)
hts["speedup_nmax"] = round(hts["full_fit"][-1] / hts["ny_fit"][-1], 0)
print(f"[Pharma-c] memory: full={hts['mem_full_gb']} GB vs nystrom={hts['mem_ny_gb']} GB")
results["hts"] = hts

with open("results_pharma.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=1)
print("\nAll pharma results saved to results_pharma.json")
