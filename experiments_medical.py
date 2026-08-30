# -*- coding: utf-8 -*-
"""
Medical applications (NEW, real data):
  A) Parkinson telemonitoring regression (UCI, 5875 records) — full GP vs Nyström
  B) Medical diagnosis classification (Cleveland heart + WDBC breast cancer) — GPC vs logistic regression
Produces: figures/fig9.png, figures/fig10.png, results_medical.json
"""
import numpy as np
import json
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import arabic_reshaper
from bidi.algorithm import get_display

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, accuracy_score,
                             roc_auc_score, log_loss, confusion_matrix)
from sklearn.model_selection import StratifiedKFold

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

C1, C2, C3, C4 = "#1f77b4", "#d62728", "#2ca02c", "#9467bd"

def rbf(X, Z, l):
    return np.exp(-cdist(X, Z, "sqeuclidean") / (2.0 * l ** 2))

def gp_fixed_predict(Xtr, ytr, Xte, ell, sig, amp=1.0):
    """full GP with fixed hyperparameters; returns mu, sd"""
    n = len(ytr)
    K = amp * rbf(Xtr, Xtr, ell) + sig ** 2 * np.eye(n)
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), ytr)
    k_ = amp * rbf(Xte, Xtr, ell)
    mu = k_ @ alpha
    v = solve_triangular(L, k_.T, lower=True)
    var = amp + sig ** 2 - np.sum(v ** 2, axis=0)
    return mu, np.sqrt(np.maximum(var, 1e-12))

def nystrom_predict(Xtr, ytr, Xm, Xte, ell, sig, amp=1.0):
    m = Xm.shape[0]
    Kmm = amp * rbf(Xm, Xm, ell) + 1e-9 * np.eye(m)
    Knm = amp * rbf(Xtr, Xm, ell)
    k_m = amp * rbf(Xte, Xm, ell)
    A = Knm.T @ Knm + sig ** 2 * Kmm
    mu = k_m @ np.linalg.solve(A, Knm.T @ ytr)
    B = Kmm + (1 / sig ** 2) * (Knm.T @ Knm)
    var = amp + sig ** 2 - np.einsum("ij,jk,ik->i", k_m, np.linalg.inv(B), k_m)
    return mu, np.sqrt(np.maximum(var, 1e-12))

def nll(y, mu, sd):
    ll = -0.5 * (np.log(2 * np.pi * sd ** 2) + ((y - mu) ** 2) / sd ** 2)
    return -float(np.mean(ll))

def timeit(fn, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out

results = {}

# ==================================================================
# A) PARKINSON TELEMONITORING
# ==================================================================
park = pd.read_csv("data/parkinsons_updrs.data")
FEAT = ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
        "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]
X_all = park[FEAT].values.astype(float)
y_all = park["motor_UPDRS"].values.astype(float)
subj = park["subject#"].values
n_all = len(y_all)
subj_ids = np.sort(np.unique(subj))

# split by subject: first 30 train, last 12 test (literature standard, no leakage)
tr_subj = subj_ids[:30]
te_subj = subj_ids[30:]
tr_mask = np.isin(subj, tr_subj)
te_mask = np.isin(subj, te_subj)
Xtr_all, ytr_all = X_all[tr_mask], y_all[tr_mask]
Xte_all, yte_all = X_all[te_mask], y_all[te_mask]

# standardize
mu_x, sd_x = Xtr_all.mean(0), Xtr_all.std(0) + 1e-12
Xtr_n = (Xtr_all - mu_x) / sd_x
Xte_n = (Xte_all - mu_x) / sd_x
mu_y, sd_y = ytr_all.mean(), ytr_all.std()
ytr_n = (ytr_all - mu_y) / sd_y
yte_n = (yte_all - mu_y) / sd_y

# hyperparameters from a small subset
r0 = np.random.default_rng(0)
idx_hp = r0.choice(len(ytr_n), 1000, replace=False)
kern = C(1.0, (0.3, 5.0)) * RBF(1.0, (2.5, 6.0)) + WhiteKernel(0.1, (1e-3, 1.0))
gpr = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=5, random_state=0)
gpr.fit(Xtr_n[idx_hp], ytr_n[idx_hp])
ELL = float(np.atleast_1d(gpr.kernel_.k1.k2.length_scale)[0])
SIG = float(np.sqrt(gpr.kernel_.k2.noise_level))
AMP = float(gpr.kernel_.k1.k1.constant_value)
print(f"[Park] hyperparams: ell={ELL:.3f} sig={SIG:.3f} amp={AMP:.3f} "
      f"| train={len(ytr_n)} test={len(yte_n)}")

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(Xtr_n, ytr_n)
pr_ridge = ridge.predict(Xte_n)
mae_ridge = float(mean_absolute_error(yte_all, pr_ridge * sd_y + mu_y))
rmse_ridge = float(np.sqrt(mean_squared_error(yte_all, pr_ridge * sd_y + mu_y)))
print(f"[Park] Ridge baseline: MAE={mae_ridge:.2f} RMSE={rmse_ridge:.2f}")
results["parkinson_ridge"] = {"mae": round(mae_ridge, 2), "rmse": round(rmse_ridge, 2)}

park_runs = {}
for n_full in [1000, 2000, 3000]:
    idx = r0.choice(len(ytr_n), n_full, replace=False)
    Xf, yf = Xtr_n[idx], ytr_n[idx]
    def fit_full():
        K = AMP * rbf(Xf, Xf, ELL) + SIG ** 2 * np.eye(n_full)
        L = cholesky(K, lower=True)
        return cho_solve((L, True), yf)
    tf, alpha = timeit(fit_full)
    def pred_full():
        k_ = AMP * rbf(Xte_n, Xf, ELL)
        return k_ @ alpha
    tp, mu_f = timeit(pred_full)
    # variance for NLL (recompute via cholesky)
    K = AMP * rbf(Xf, Xf, ELL) + SIG ** 2 * np.eye(n_full)
    L = cholesky(K, lower=True)
    k_ = AMP * rbf(Xte_n, Xf, ELL)
    v = solve_triangular(L, k_.T, lower=True)
    sd_f = np.sqrt(np.maximum(AMP + SIG ** 2 - np.sum(v ** 2, axis=0), 1e-12))
    mae_f = float(mean_absolute_error(yte_all, mu_f * sd_y + mu_y))
    rmse_f = float(np.sqrt(mean_squared_error(yte_all, mu_f * sd_y + mu_y)))
    nll_f = nll(yte_n, mu_f, sd_f)
    park_runs[f"full_{n_full}"] = {"time_fit": round(tf, 3), "time_pred": round(tp, 3),
                                   "mae": round(mae_f, 2), "rmse": round(rmse_f, 2),
                                   "nll": round(nll_f, 3)}
    print(f"[Park] full n={n_full}: fit={tf:.2f}s pred={tp:.2f}s MAE={mae_f:.2f} "
          f"RMSE={rmse_f:.2f} NLL={nll_f:.3f}")

for m in [100, 200]:
    idx_m = r0.choice(len(ytr_n), m, replace=False)
    Xm = Xtr_n[idx_m]
    def fit_ny():
        Kmm = AMP * rbf(Xm, Xm, ELL) + 1e-9 * np.eye(m)
        Knm = AMP * rbf(Xtr_n, Xm, ELL)
        A = Knm.T @ Knm + SIG ** 2 * Kmm
        return np.linalg.solve(A, Knm.T @ ytr_n)
    tnf, a_ny = timeit(fit_ny)
    def pred_ny():
        k_m = AMP * rbf(Xte_n, Xm, ELL)
        return k_m @ a_ny
    tnp, mu_n = timeit(pred_ny)
    mu_n_all, sd_n_all = nystrom_predict(Xtr_n, ytr_n, Xm, Xte_n, ELL, SIG, AMP)
    mae_n = float(mean_absolute_error(yte_all, mu_n_all * sd_y + mu_y))
    rmse_n = float(np.sqrt(mean_squared_error(yte_all, mu_n_all * sd_y + mu_y)))
    nll_n = nll(yte_n, mu_n_all, sd_n_all)
    park_runs[f"nystrom_{m}"] = {"time_fit": round(tnf, 4), "time_pred": round(tnp, 4),
                                 "mae": round(mae_n, 2), "rmse": round(rmse_n, 2),
                                 "nll": round(nll_n, 3)}
    print(f"[Park] nystrom m={m}: fit={tnf:.4f}s pred={tnp:.4f}s MAE={mae_n:.2f} "
          f"RMSE={rmse_n:.2f} NLL={nll_n:.3f}")

n_tr_park = len(ytr_n)
results["parkinson"] = {
    "n_total": int(n_all), "n_train": int(n_tr_park), "n_test": int(len(yte_n)),
    "n_subjects": int(len(subj_ids)), "ell": round(ELL, 3), "sig": round(SIG, 3),
    "amp": round(AMP, 3), "runs": park_runs,
    "ridge": {"mae": round(mae_ridge, 2), "rmse": round(rmse_ridge, 2)},
    "mem_full_mb": round(n_tr_park ** 2 * 8 / 1e6, 0),
    "mem_ny200_mb": round((n_tr_park * 200 + 200 ** 2) * 8 / 1e6, 1),
}

# ---------- fig9: Parkinson ----------
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
ax = axes[0]
ax.scatter(yte_all, mu_n_all * sd_y + mu_y, s=14, c=C1, alpha=0.5)
lims = [min(yte_all.min(), (mu_n_all * sd_y + mu_y).min()) - 2,
        max(yte_all.max(), (mu_n_all * sd_y + mu_y).max()) + 2]
ax.plot(lims, lims, "--", color=C2, lw=1.5, label=fa("خط y=x"))
ax.set_xlabel(fa("امتیاز واقعی motor-UPDRS"))
ax.set_ylabel(fa("امتیاز پیش‌بینی‌شده"))
ax.set_title(fa(f"(الف) پراکندگی پیش‌بینی (نیستروم m=۲۰۰) — MAE={park_runs['nystrom_200']['mae']}"),
             fontsize=10.5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
# (b) trajectory of one test subject
ax = axes[1]
te_subj_list = np.unique(subj[te_mask])
subj_show = None
for s in te_subj_list:
    msk = subj == s
    if msk.sum() >= 25:
        subj_show = s
        break
if subj_show is None:
    subj_show = te_subj_list[0]
msk = subj == subj_show
X_s = (X_all[msk] - mu_x) / sd_x
y_s = y_all[msk]
t_s = park["test_time"].values[msk]
ord_ = np.argsort(t_s)
mu_s, sd_s = nystrom_predict(Xtr_n, ytr_n, Xm, X_s, ELL, SIG, AMP)
mu_s = mu_s * sd_y + mu_y
sd_s = sd_s * sd_y
ax.plot(t_s[ord_], y_s[ord_], "o-", color=C2, ms=4, lw=1.2, label=fa("مقادیر واقعی"))
ax.plot(t_s[ord_], mu_s[ord_], "-", color=C1, lw=1.8, label=fa("پیش‌بینی GP (نیستروم)"))
ax.fill_between(t_s[ord_], mu_s[ord_] - 2 * sd_s[ord_], mu_s[ord_] + 2 * sd_s[ord_],
                color=C1, alpha=0.15, label=fa("فاصله اطمینان ۹۵٪"))
ax.set_xlabel(fa("زمان سنجش (روز)"))
ax.set_ylabel(fa("امتیاز motor-UPDRS"))
ax.set_title(fa(f"(ب) روند پیشرفت بیماری — بیمار شماره {fad(str(int(subj_show)))}"), fontsize=10.5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.suptitle(fa("پیش‌بینی شدت پارکینسون از ویژگی‌های صوتی با فرآیند گوسی"), fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig9.png", dpi=200)
plt.close(fig)
print("[fig] fig9.png saved")

# ==================================================================
# B) CLASSIFICATION — Cleveland heart + WDBC
# ==================================================================
def run_clf(name, X, y, seeds=(0, 1, 2, 3, 4)):
    res = {"gpc": {"acc": [], "auc": [], "ll": []},
           "lr": {"acc": [], "auc": [], "ll": []}}
    best = None
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(X, y):
            Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
            mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-12
            Xtr_s, Xte_s = (Xtr - mu_x) / sd_x, (Xte - mu_x) / sd_x
            gpc = GaussianProcessClassifier(kernel=C(1.0) * RBF(1.0),
                                            n_restarts_optimizer=3, random_state=seed)
            gpc.fit(Xtr_s, ytr)
            pg = gpc.predict_proba(Xte_s)[:, 1]
            yg = gpc.predict(Xte_s)
            lr = LogisticRegression(max_iter=2000)
            lr.fit(Xtr_s, ytr)
            pl = lr.predict_proba(Xte_s)[:, 1]
            yl = lr.predict(Xte_s)
            res["gpc"]["acc"].append(accuracy_score(yte, yg))
            res["gpc"]["auc"].append(roc_auc_score(yte, pg))
            res["gpc"]["ll"].append(log_loss(yte, pg))
            res["lr"]["acc"].append(accuracy_score(yte, yl))
            res["lr"]["auc"].append(roc_auc_score(yte, pl))
            res["lr"]["ll"].append(log_loss(yte, pl))
            if best is None:
                best = dict(Xte=Xte_s, yte=yte, pg=pg, pl=pl, yg=yg, yl=yl)
    out = {"gpc": {k: {"mean": round(float(np.mean(res["gpc"][k])), 3),
                       "std": round(float(np.std(res["gpc"][k])), 3)} for k in ("acc", "auc", "ll")},
           "lr": {k: {"mean": round(float(np.mean(res["lr"][k])), 3),
                      "std": round(float(np.std(res["lr"][k])), 3)} for k in ("acc", "auc", "ll")}}
    print(f"[Clf-{name}] GPC acc={out['gpc']['acc']['mean']}±{out['gpc']['acc']['std']} "
          f"auc={out['gpc']['auc']['mean']} ll={out['gpc']['ll']['mean']} | "
          f"LR acc={out['lr']['acc']['mean']}±{out['lr']['acc']['std']} "
          f"auc={out['lr']['auc']['mean']} ll={out['lr']['ll']['mean']}")
    return out, best

# heart (Cleveland)
cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
heart = pd.read_csv("data/heart_cleveland.data", header=None, names=cols, na_values="?")
for c in heart.columns:
    heart[c] = heart[c].fillna(heart[c].median())
Xh = heart.drop(columns=["num"]).values.astype(float)
yh = (heart["num"].values > 0).astype(int)
res_heart, best_heart = run_clf("Heart", Xh, yh)
cm_h = confusion_matrix(best_heart["yte"], best_heart["yg"])

# WDBC
wdbc = pd.read_csv("data/wdbc.data", header=None)
Xw = wdbc.iloc[:, 2:].values.astype(float)
yw = (wdbc.iloc[:, 1].values == "M").astype(int)
res_wdbc, best_wdbc = run_clf("WDBC", Xw, yw)
cm_w = confusion_matrix(best_wdbc["yte"], best_wdbc["yw"] if False else best_wdbc["yg"])

results["heart"] = {"n": int(len(yh)), "n_features": int(Xh.shape[1]),
                    "pos_rate": round(float(yh.mean()), 3), **res_heart,
                    "cm_gpc": cm_h.tolist()}
results["wdbc"] = {"n": int(len(yw)), "n_features": int(Xw.shape[1]),
                   "pos_rate": round(float(yw.mean()), 3), **res_wdbc,
                   "cm_gpc": cm_w.tolist()}

# ---------- fig10: classification ----------
from sklearn.metrics import roc_curve, auc
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))

def draw_cm(ax, cm, title):
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, fad(str(cm[i, j])), ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels([fa("سالم (۰)"), fa("بیمار (۱)")])
    ax.set_yticks([0, 1]); ax.set_yticklabels([fa("سالم (۰)"), fa("بیمار (۱)")])
    ax.set_xlabel(fa("پیش‌بینی")); ax.set_ylabel(fa("واقعی"))
    ax.set_title(title, fontsize=10.5)

def draw_roc(ax, yte, pg, pl, title):
    fpr_g, tpr_g, _ = roc_curve(yte, pg)
    fpr_l, tpr_l, _ = roc_curve(yte, pl)
    ax.plot(fpr_g, tpr_g, "-", color=C1, lw=2,
            label=fa(f"GPC (AUC={auc(fpr_g, tpr_g):.3f})"))
    ax.plot(fpr_l, tpr_l, "--", color=C2, lw=2,
            label=fa(f"رگرسیون لجستیک (AUC={auc(fpr_l, tpr_l):.3f})"))
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel(fa("نرخ مثبت کاذب")); ax.set_ylabel(fa("نرخ مثبت حقیقی"))
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

draw_cm(axes[0, 0], cm_h, fa("(الف) ماتریس درهم‌ریختگی — بیماری قلبی (کلیولند)"))
draw_roc(axes[0, 1], best_heart["yte"], best_heart["pg"], best_heart["pl"],
         fa("(ب) منحنی ROC — بیماری قلبی"))
draw_cm(axes[1, 0], cm_w, fa("(ج) ماتریس درهم‌ریختگی — سرطان پستان (WDBC)"))
draw_roc(axes[1, 1], best_wdbc["yte"], best_wdbc["pg"], best_wdbc["pl"],
         fa("(د) منحنی ROC — سرطان پستان"))
fig.suptitle(fa("طبقه‌بندی پزشکی با فرآیند گوسی در برابر رگرسیون لجستیک"), fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig10.png", dpi=200)
plt.close(fig)
print("[fig] fig10.png saved")

with open("results_medical.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=1)
print("\nAll medical results saved to results_medical.json")
