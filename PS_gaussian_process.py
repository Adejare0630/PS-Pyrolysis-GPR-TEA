#%% [markdown]
# Gaussian process regression (GPR) for PS pyrolysis yields
#
# Recommended setup from the feature/target search:
#   inputs  : temperature + regime class + catalyst class + study (one-hot "study effect")
#   outputs : liquid, styrene, BTX, total aromatics (wt% of feed) - one GPR per output
#   kernel  : constant x anisotropic RBF (one length scale per input) + white noise
#
# Run cell by cell in VS Code (Python + Jupyter extensions) or:  python PS_gaussian_process.py
# Needs   : pandas, numpy, scikit-learn, matplotlib, openpyxl, joblib
# Data    : PS_ML_training_table_1.xlsx (sheet "Training table") in the same folder as this file
# Scope   : pure PS at normal pressure - HIPS feeds, PS/polymer blends and the vacuum run are excluded

#%% Imports and settings
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
# folder of this script, also in the VS Code interactive window / Jupyter (where __file__ is not defined and the
# working folder can be a different one, e.g. Downloads, holding older data or model files)
HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or globals().get("__vsc_ipynb_file__")
                                       or os.path.join(os.getcwd(), "_")))
DATA = os.path.join(HERE, "PS_ML_training_table_1.xlsx")
SEED = 42

TARGETS = {"Liquid": "Liquid (wt% of feed)",
           "Styrene": "Styrene (wt% of feed)",
           "BTX": "BTX (wt% of feed)",
           "Total aromatics": "Total aromatics (wt% of feed)"}
# TARGETS["Gas"] = "Gas (wt% of feed)"          # uncomment to add gas

#%% Load data
df = pd.read_excel(DATA, sheet_name="Training table", header=1)
df = df.rename(columns={"Temperature (°C)": "T", "Regime class": "regime", "Catalyst class": "catalyst",
                        "Study ID (group for CV)": "study_id"})
df["T"] = pd.to_numeric(df["T"], errors="coerce")
for col in TARGETS.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep the dataset to pure PS at normal pressure: drop HIPS feeds, PS/polymer blends and the vacuum run
excl = (df["Feed description"].astype(str).str.contains("HIPS")
        | (pd.to_numeric(df["PS (wt% of feed)"], errors="coerce") < 100)
        | df["Pressure as reported"].astype(str).str.fullmatch("Vacuum"))
excluded = df.loc[excl, ["Study", "Run / condition", "Feed description", "PS (wt% of feed)", "Pressure as reported"]].copy()
excluded["Reason"] = np.where(excluded["Feed description"].astype(str).str.contains("HIPS"), "HIPS feed (not PS)",
                     np.where(pd.to_numeric(excluded["PS (wt% of feed)"], errors="coerce") < 100, "Mixed-plastic feed (PS < 100 wt%)",
                              "Vacuum operation"))
df = df[~excl].reset_index(drop=True)
print(f"Excluded {excl.sum()} runs:"); print(excluded[["Study", "Run / condition", "Reason"]].to_string(index=False))

REGIMES = sorted(df["regime"].dropna().unique())
CATALYSTS = sorted(df["catalyst"].dropna().unique())
STUDIES = sorted(df["study_id"].dropna().unique())
STUDY_NAMES = df.groupby("study_id")["Study"].first().to_dict()


def make_features(frame):
    """Temperature + one-hot regime + one-hot catalyst + one-hot study."""
    X = pd.DataFrame({"T": frame["T"].astype(float).values}, index=frame.index)
    for r in REGIMES:
        X[f"regime: {r}"] = (frame["regime"] == r).astype(float)
    for c in CATALYSTS:
        X[f"catalyst: {c}"] = (frame["catalyst"] == c).astype(float)
    for s in STUDIES:
        X[f"study {s}"] = (frame["study_id"] == s).astype(float)
    return X


#%% The model
def make_gpr(n_features):
    # length scales bounded at >= 1.5 standard deviations (~100 °C for temperature) so the
    # trends stay smooth instead of chasing single points
    kernel = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=[3.0] * n_features, length_scale_bounds=(1.5, 1e4))
              + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-4, 10)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=SEED)
    return make_pipeline(StandardScaler(), gpr)


#%% Cross-validation (repeated 5-fold) for each output
cv_rows, oof = [], {}
for name, col in TARGETS.items():
    d = df.dropna(subset=["T", "regime", "catalyst", col]).reset_index(drop=True)
    X, y = make_features(d).values, d[col].values
    pred = np.full((3, len(y)), np.nan)
    r2s, rmses, maes = [], [], []
    for k, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X)):
        m = make_gpr(X.shape[1]).fit(X[tr], y[tr])
        p = m.predict(X[te]); pred[k // 5, te] = p
        r2s.append(r2_score(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))
    oof[name] = (y, pred[0])
    cv_rows.append({"Output": name, "Runs": len(y), "Studies": d["study_id"].nunique(),
                    "R2": np.mean(r2s), "R2 SD": np.std(r2s),
                    "RMSE (wt%)": np.mean(rmses), "MAE (wt%)": np.mean(maes)})
    print(f"{name:16s} R2 = {np.mean(r2s):.2f} ± {np.std(r2s):.2f}   RMSE = {np.mean(rmses):.2f}   MAE = {np.mean(maes):.2f}")
cv_table = pd.DataFrame(cv_rows)

#%% Final models on all data
models, train_fit = {}, {}
for name, col in TARGETS.items():
    d = df.dropna(subset=["T", "regime", "catalyst", col]).reset_index(drop=True)
    models[name] = make_gpr(len(make_features(d).columns)).fit(make_features(d).values, d[col].values)
    print(f"{name}: fitted kernel = {models[name][-1].kernel_}")
    p_tr = models[name].predict(make_features(d).values); y_tr = d[col].values
    train_fit[name] = (y_tr, p_tr)
    i = cv_table.index[cv_table["Output"] == name][0]
    cv_table.loc[i, "Training R2"] = r2_score(y_tr, p_tr)
    cv_table.loc[i, "Training RMSE (wt%)"] = np.sqrt(mean_squared_error(y_tr, p_tr))
    cv_table.loc[i, "Training MAE (wt%)"] = mean_absolute_error(y_tr, p_tr)
cv_table = cv_table.rename(columns={"R2": "Test R2 (5-fold CV)", "R2 SD": "Test R2 SD",
                                    "RMSE (wt%)": "Test RMSE (wt%)", "MAE (wt%)": "Test MAE (wt%)"})
cv_table["R2 gap (train - test)"] = cv_table["Training R2"] - cv_table["Test R2 (5-fold CV)"]
print("\nTraining vs test performance\n", cv_table.round(2).to_string(index=False))
joblib.dump({"models": models, "regimes": REGIMES, "catalysts": CATALYSTS, "studies": STUDIES,
             "targets": list(TARGETS)}, os.path.join(HERE, "PS_gaussian_process.joblib"))


#%% Prediction
def predict(T, regime="continuous", catalyst="none", study_id=None):
    """
    Predict yields (wt% of feed) with a 95% interval.
    study_id given  -> prediction for that study's reactor/lab (its study effect is used).
    study_id = None -> new process: the prediction is averaged over all studies in the data, and the
                       interval includes the spread between studies.
    """
    T = np.atleast_1d(np.asarray(T, float))
    out = pd.DataFrame({"Temperature (°C)": T})
    studies = [study_id] if study_id is not None else STUDIES
    for name, m in models.items():
        mus, sds = [], []
        for s in studies:
            frame = pd.DataFrame({"T": T, "regime": regime, "catalyst": catalyst, "study_id": s})
            mu, sd = m[-1].predict(m[0].transform(make_features(frame).values), return_std=True)
            mus.append(mu); sds.append(sd)
        mus, sds = np.array(mus), np.array(sds)
        mean = mus.mean(0)
        sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))            # within + between-study variance
        out[f"{name} (wt%)"] = np.clip(mean, 0, 100)
        out[f"{name} 95% low"] = np.clip(mean - 1.96 * sd, 0, 100)
        out[f"{name} 95% high"] = np.clip(mean + 1.96 * sd, 0, 100)
    return out


# examples
print(predict([450, 500, 550, 600], regime="continuous", catalyst="none").round(1).to_string(index=False))
print(predict([450, 500, 550], regime="continuous", catalyst="none", study_id=STUDIES[0]).round(1).to_string(index=False))

#%% Figure: observed vs cross-validated prediction
# Figures are drawn at their printed size (<= 7.2 in wide) with 8-9 pt text, so they stay readable in the journal
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight"})
hi_ok = lambda y, p: max(y.max(), np.nanmax(p)) > 20
fig, axes = plt.subplots(2, 2, figsize=(6.2, 6.4))
for ax, (name, (y, p)) in zip(np.atleast_1d(axes).ravel(), oof.items()):
    lo = 0 if min(y.min(), np.nanmin(p)) < 20 else 10 * np.floor(min(y.min(), np.nanmin(p)) / 10 - 0.5)   # axes fit the data
    hi = 10 * np.ceil(max(y.max(), np.nanmax(p)) / 10) if hi_ok(y, p) else max(y.max(), np.nanmax(p)) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
    ax.scatter(y, p, s=14, color="#2a6fb0", edgecolor="white", linewidth=0.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    row = cv_table.set_index("Output").loc[name]
    ax.set_title(f"{name}\nR² = {row['Test R2 (5-fold CV)']:.2f} ± {row['Test R2 SD']:.2f}, RMSE = {row['Test RMSE (wt%)']:.1f}, MAE = {row['Test MAE (wt%)']:.1f} wt%")
    ax.set_xlabel("Observed (wt% of feed)"); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
for ax in np.atleast_1d(axes)[:, 0]: ax.set_ylabel("Predicted, 5-fold CV (wt% of feed)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_GPR_parity.png")); plt.show()

#%% Figure: training fit vs cross-validated test predictions
fig, axes = plt.subplots(2, len(TARGETS), figsize=(7.2, 4.4))
tab = cv_table.set_index("Output")
for j, name in enumerate(TARGETS):
    for r, (label, (y, p), key) in enumerate([("Training", train_fit[name], "Training"),
                                               ("CV test", oof[name], "Test")]):
        ax = axes[r, j]
        lo = 0 if min(y.min(), np.nanmin(p)) < 20 else 10 * np.floor(min(y.min(), np.nanmin(p)) / 10 - 0.5)
        hi = 10 * np.ceil(max(y.max(), np.nanmax(p)) / 10)
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
        ax.scatter(y, p, s=8, color="#7a7a7a" if r == 0 else "#2a6fb0", edgecolor="white", linewidth=0.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        r2c = "Training R2" if r == 0 else "Test R2 (5-fold CV)"
        rmc = "Training RMSE (wt%)" if r == 0 else "Test RMSE (wt%)"
        mac = "Training MAE (wt%)" if r == 0 else "Test MAE (wt%)"
        ax.set_title(f"{name}\nR² {tab.loc[name, r2c]:.2f}, RMSE {tab.loc[name, rmc]:.1f}", fontsize=8)
        ax.tick_params(labelsize=7.5)
        if r == 1: ax.set_xlabel("Observed (wt%)")
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        if j == 0: ax.set_ylabel(f"{label}\npredicted (wt%)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_GPR_train_vs_test.png")); plt.show()

#%% Figure: temperature trends for a new process (95% interval)
temps = np.linspace(350, 650, 61)
fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.2, 2.3), sharex=True)
for cat, color in [("none", "#2a6fb0"), ("basic", "#d9822b")]:
    if cat not in CATALYSTS: continue
    res = predict(temps, regime="continuous" if cat == "none" else "batch (slow)", catalyst=cat)
    for ax, name in zip(np.atleast_1d(axes), TARGETS):
        ax.plot(temps, res[f"{name} (wt%)"], color=color, lw=2, label=f"catalyst: {cat}")
        ax.fill_between(temps, res[f"{name} 95% low"], res[f"{name} 95% high"], color=color, alpha=0.12)
for ax, (name, col) in zip(np.atleast_1d(axes), TARGETS.items()):
    ax.scatter(df["T"], df[col], s=8, color="#b0b0b0", zorder=0)
    ax.set_title(name); ax.set_xlabel("Temperature (°C)"); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("Yield (wt% of feed)"); np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_GPR_temperature_trends.png")); plt.show()

#%% Save results
with pd.ExcelWriter(os.path.join(HERE, "PS_gaussian_process_results.xlsx")) as xw:
    cv_table.round(3).to_excel(xw, sheet_name="Train vs test performance", index=False)
    excluded.to_excel(xw, sheet_name="Excluded runs", index=False)
    grid = []
    for reg in REGIMES:
        for cat in CATALYSTS:
            g = predict(np.arange(350, 651, 25), regime=reg, catalyst=cat); g.insert(0, "Catalyst", cat); g.insert(0, "Regime", reg)
            grid.append(g)
    pd.concat(grid).round(2).to_excel(xw, sheet_name="New-process predictions", index=False)
    pd.DataFrame({"Study ID": STUDIES, "Study": [STUDY_NAMES[s] for s in STUDIES]}).to_excel(xw, sheet_name="Study IDs", index=False)
print("Saved model, results and figures to", HERE)
