#%% [markdown]
# Diagnostic plots for the PS pyrolysis Gaussian process (GPR) model
#
# Shows (a) that the model has learned properly and (b) that it is fit to feed the process simulation.
#   1  Residuals vs predicted and vs temperature          7  Within-study temperature trends
#   2  Residual histogram and Q-Q plot                     8  Yield-temperature curves with 95% bands
#   3  Learning curve (training-set size)                  9  Mass-balance consistency
#   4  Interval calibration                               10  Range-of-validity map
#   5  Model comparison (GPR vs linear, RF, DeepNN)       11  GPR vs process-model (Aspen) stream yields
#   6  What drives the predictions (grouped permutation importance)
#
# Run cell by cell in VS Code, or:  python PS_GPR_diagnostics.py
# Needs PS_gaussian_process.py and PS_ML_training_table_1.xlsx in the same folder.
# Figures are saved as fig_diag_01 ... fig_diag_11 (PNG) next to this file. Runtime ~5-8 min.

#%% Load data, features and model definition from PS_gaussian_process.py (no training is repeated there)
import os
# folder of this script, also in the VS Code interactive window / Jupyter (where __file__ is not defined and the
# working folder can be a different one, e.g. Downloads, holding older data or model files)
HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or globals().get("__vsc_ipynb_file__")
                                       or os.path.join(os.getcwd(), "_")))
_src = open(os.path.join(HERE, "PS_gaussian_process.py"), encoding="utf-8").read()
exec(_src.split("#%% Cross-validation")[0])          # defines df, TARGETS, make_features, make_gpr, ...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (validated palette)
GREY = "#8a8a8a"
# drawn at printed size (<= 7.2 in wide) with 7.5-8.5 pt text; the figure captions carry the titles
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
NAMES = list(TARGETS)


def save(fig, n, title):
    print(f"fig_diag_{n:02d}: {title}")                 # the title goes in the caption, not on the figure
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"fig_diag_{n:02d}.png"), dpi=300, bbox_inches="tight")
    plt.show()


def gp_predict(model, X, return_std=False):
    """Prediction (and standard deviation) from a StandardScaler + GPR pipeline."""
    return model[-1].predict(model[0].transform(X), return_std=return_std)


def data_for(name):
    d = df.dropna(subset=["T", "regime", "catalyst", TARGETS[name]]).reset_index(drop=True)
    return d, make_features(d), d[TARGETS[name]].values


#%% Core run: 5-fold cross-validation with predictive SD, kept for plots 1, 2, 4 and 6
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv = {}
for name in NAMES:
    d, Xdf, y = data_for(name)
    X = Xdf.values
    mu, sd, fold_models = np.zeros_like(y), np.zeros_like(y), []
    for tr, te in kf.split(X):
        m = make_gpr(X.shape[1]).fit(X[tr], y[tr])
        mu[te], sd[te] = gp_predict(m, X[te], return_std=True)
        fold_models.append((m, tr, te))
    cv[name] = dict(d=d, Xdf=Xdf, y=y, mu=mu, sd=sd, folds=fold_models)
    print(f"{name:16s} 5-fold R2 = {r2_score(y, mu):.2f}")

final = {}
for name in NAMES:
    d, Xdf, y = data_for(name)
    final[name] = make_gpr(Xdf.shape[1]).fit(Xdf.values, y)


#%% Plot 1 - Residuals vs predicted and vs temperature
def binned_mean(x, r, nbins=6):
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    xc, rm = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() >= 3: xc.append(x[m].mean()); rm.append(r[m].mean())
    return np.array(xc), np.array(rm)

fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2))
for j, name in enumerate(NAMES):
    c = cv[name]; res = c["y"] - c["mu"]
    for i, (x, xl) in enumerate([(c["mu"], "Predicted (wt%)"), (c["d"]["T"].values, "Temperature (°C)")]):
        ax = axes[i, j]
        ax.axhline(0, color="black", lw=0.8)
        ax.scatter(x, res, s=7, color=COLORS[0], edgecolor="white", linewidth=0.3, alpha=0.85)
        xb, rb = binned_mean(x, res)
        ax.plot(xb, rb, color=COLORS[1], lw=1.5, marker="o", ms=3, label="Binned mean")
        ax.set_xlabel(xl); ax.set_title(name)
        if j == 0: ax.set_ylabel("Residual (wt%)")
axes[0, 0].legend(frameon=False)
save(fig, 1, "Plot 1 - Residuals (5-fold CV): no trend with predicted yield or temperature means no systematic bias")


#%% Plot 2 - Residual distribution: histogram and Q-Q plot of standardised residuals
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4))
for j, name in enumerate(NAMES):
    c = cv[name]; z = (c["y"] - c["mu"]) / c["sd"]
    ax = axes[0, j]
    ax.hist(z, bins=20, density=True, color=COLORS[0], edgecolor="white")
    zz = np.linspace(-4, 4, 200); ax.plot(zz, stats.norm.pdf(zz), color=COLORS[1], lw=2, label="Standard normal")
    ax.set_xlim(-4.5, 4.5); ax.set_title(f"{name}\nmean {z.mean():.2f}, SD {z.std():.2f}")
    ax.set_xlabel("Standardised residual")
    if j == 0: ax.set_ylabel("Density"); ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    ax = axes[1, j]
    (osm, osr), (slope, icpt, rr) = stats.probplot(z, dist="norm")
    ax.scatter(osm, osr, s=7, color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.plot(osm, osm, "--", color=GREY, lw=1)
    ax.set_xlabel("Normal quantile"); ax.set_title(f"Q-Q, r = {rr:.2f}")
    if j == 0: ax.set_ylabel("Ordered std. residual")
save(fig, 2, "Plot 2 - Residual distribution: centred near 0 with SD near 1 means errors and stated uncertainty are consistent")


#%% Plot 3 - Learning curve
fractions = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
rng = np.random.default_rng(SEED)
lc = []
for name in NAMES:
    d, Xdf, y = data_for(name); X = Xdf.values
    for tr, te in kf.split(X):
        for f in fractions:
            sub = rng.choice(tr, size=max(10, int(round(f * len(tr)))), replace=False)
            m = make_gpr(X.shape[1]).fit(X[sub], y[sub])
            lc.append(dict(Output=name, fraction=f, n_train=len(sub),
                           train=np.sqrt(mean_squared_error(y[sub], gp_predict(m, X[sub]))),
                           test=np.sqrt(mean_squared_error(y[te], gp_predict(m, X[te])))))
lc = pd.DataFrame(lc)
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
for ax, name in zip(axes, NAMES):
    g = lc[lc.Output == name].groupby("fraction").agg(n=("n_train", "mean"), tr=("train", "mean"), trs=("train", "std"),
                                                     te=("test", "mean"), tes=("test", "std"))
    for col, sd_, colr, lab in [("tr", "trs", GREY, "Training"), ("te", "tes", COLORS[0], "Test")]:
        ax.plot(g["n"], g[col], marker="o", ms=3, color=colr, lw=1.5, label=lab)
        ax.fill_between(g["n"], g[col] - g[sd_], g[col] + g[sd_], color=colr, alpha=0.15)
    ax.set_title(name); ax.set_xlabel("Training runs"); ax.set_ylim(bottom=0)
axes[0].set_ylabel("RMSE (wt%)"); axes[0].legend(frameon=False)
save(fig, 3, "Plot 3 - Learning curve: test error falls and the train-test gap narrows as data are added")


#%% Plot 4 - Interval calibration
levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
fig, ax = plt.subplots(figsize=(3.5, 3.4))
ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Perfect calibration")
for k, name in enumerate(NAMES):
    c = cv[name]; z = np.abs(c["y"] - c["mu"]) / c["sd"]
    obs = [(z <= stats.norm.ppf(0.5 + l / 2)).mean() for l in levels]
    ax.plot(levels, obs, marker="o", ms=3, lw=1.5, color=COLORS[k], label=f"{name} (95%: {obs[9]:.2f})")
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=7, loc="upper left")
save(fig, 4, "Plot 4 - Interval calibration (5-fold CV)")


#%% Plot 5 - Model comparison: GPR vs ridge, random forest and DeepNN (same inputs, same folds)
def deepnn():   # the manuscript architecture: 64-64 ReLU, Adam lr 0.002, 1000 iterations, no regularisation
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam", learning_rate_init=0.002,
                       max_iter=1000, alpha=0.0, random_state=SEED)
    return TransformedTargetRegressor(make_pipeline(StandardScaler(), mlp), transformer=StandardScaler())

others = {"Linear (ridge)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
          "Random forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED),
          "DeepNN (64-64)": deepnn}
cmp_rows = []
for name in NAMES:
    c = cv[name]; X, y = c["Xdf"].values, c["y"]
    tr_r2 = np.mean([r2_score(y[tr], gp_predict(m, X[tr])) for m, tr, te in c["folds"]])
    cmp_rows.append(dict(Output=name, Model="GPR", Train=tr_r2, Test=r2_score(y, c["mu"])))
    for mn, f in others.items():
        P, trs = np.zeros_like(y), []
        for tr, te in kf.split(X):
            m = f().fit(X[tr], y[tr]); P[te] = m.predict(X[te]); trs.append(r2_score(y[tr], m.predict(X[tr])))
        cmp_rows.append(dict(Output=name, Model=mn, Train=np.mean(trs), Test=r2_score(y, P)))
cmp = pd.DataFrame(cmp_rows)
models_order = ["GPR", "Linear (ridge)", "Random forest", "DeepNN (64-64)"]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
w = 0.38
for ax, name in zip(axes, NAMES):
    s = cmp[cmp.Output == name].set_index("Model").loc[models_order]
    x = np.arange(len(models_order))
    ax.bar(x - w / 2, s["Train"], w, color="#c9c9c9", label="Training R²")
    ax.bar(x + w / 2, s["Test"].clip(lower=-0.5), w, color=COLORS[0], label="Test R² (5-fold CV)")
    for xi, v in zip(x, s["Test"]):
        ax.text(xi + w / 2, max(v, -0.5) + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(["GPR", "Ridge", "RF", "DeepNN"], rotation=35, ha="right"); ax.set_title(name)
    ax.axhline(0, color="black", lw=0.8); ax.set_ylim(-0.5, 1.05)
axes[0].set_ylabel("R²")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.06))
save(fig, 5, "Plot 5 - Model comparison on identical inputs and folds (test bars below -0.5 are clipped; value printed)")


#%% Plot 6 - What drives the predictions: grouped permutation importance on held-out folds
groups = {"Temperature": ["T"],
          "Regime": [c for c in cv[NAMES[0]]["Xdf"].columns if c.startswith("regime:")],
          "Catalyst": [c for c in cv[NAMES[0]]["Xdf"].columns if c.startswith("catalyst:")],
          "Study effect": [c for c in cv[NAMES[0]]["Xdf"].columns if c.startswith("study ")]}
imp = []
for name in NAMES:
    c = cv[name]; Xdf, y = c["Xdf"], c["y"]
    base = r2_score(y, c["mu"])
    for gname, cols in groups.items():
        idx = [Xdf.columns.get_loc(k) for k in cols if k in Xdf.columns]
        drops = []
        for rep in range(10):
            P = np.zeros_like(y)
            for m, tr, te in c["folds"]:
                Xt = Xdf.values[te].copy()
                perm = np.random.default_rng(rep).permutation(len(te))
                Xt[:, idx] = Xt[perm][:, idx]               # shuffle the whole group together
                P[te] = gp_predict(m, Xt)
            drops.append(base - r2_score(y, P))
        imp.append(dict(Output=name, Group=gname, drop=np.mean(drops), sd=np.std(drops)))
imp = pd.DataFrame(imp)
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharex=True, sharey=True)
for ax, name in zip(axes, NAMES):
    s = imp[imp.Output == name].set_index("Group").loc[list(groups)]
    ax.barh(list(groups), s["drop"], xerr=s["sd"], color=COLORS[0], ecolor=GREY)
    ax.set_title(name); ax.set_xlabel("Drop in test R²")
axes[0].invert_yaxis()
save(fig, 6, "Plot 6 - What drives the predictions (grouped permutation importance, 5-fold CV, 10 repeats)")


#%% Plot 7 - Within-study temperature trends: model curve vs each study's data
trend_studies = [s for s in ["Gentilcore et al. 2025", "Gonzalez-Aguilar et al. 2022", "Inayat et al. 2022", "Albor et al. 2023"]
                 if s in set(df["Study"])]
fig, axes = plt.subplots(len(trend_studies), 4, figsize=(7.2, 1.85 * len(trend_studies) + 0.3), squeeze=False)
for i, st in enumerate(trend_studies):
    ds = df[df["Study"] == st]
    sid = ds["study_id"].iloc[0]
    combo = ds.groupby(["regime", "catalyst"]).size().idxmax()           # the study's main condition
    dsc = ds[(ds.regime == combo[0]) & (ds.catalyst == combo[1])]
    tgrid = np.linspace(dsc["T"].min() - 15, dsc["T"].max() + 15, 60)
    for j, name in enumerate(NAMES):
        ax = axes[i, j]; col = TARGETS[name]
        frame = pd.DataFrame({"T": tgrid, "regime": combo[0], "catalyst": combo[1], "study_id": sid})
        mu, sd = gp_predict(final[name], make_features(frame).values, return_std=True)
        ax.plot(tgrid, mu, color=COLORS[0], lw=1.5, label="GPR (this study)")
        ax.fill_between(tgrid, mu - 1.96 * sd, mu + 1.96 * sd, color=COLORS[0], alpha=0.15)
        ok = dsc[col].notna()
        ax.scatter(dsc.loc[ok, "T"], dsc.loc[ok, col], s=12, color=COLORS[1], edgecolor="white", linewidth=0.4, zorder=3,
                   label="Observed")
        if i == 0: ax.set_title(name)
        if i == len(trend_studies) - 1: ax.set_xlabel("Temperature (°C)")
        if j == 0: ax.set_ylabel(f"{st.split(' et al')[0]}\n{combo[0]}, {combo[1]}\nyield (wt%)", fontsize=7.5)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.03))
save(fig, 7, "Plot 7 - Within-study trends: the fitted curve follows each study's own temperature response")


#%% Plot 8 - Yield-temperature curves for a new process with 95% bands (simulation input)
def predict_new(T, regime, catalyst, name):
    T = np.asarray(T, float); mus, sds = [], []
    for s in STUDIES:                                         # average over studies = "new process"
        frame = pd.DataFrame({"T": T, "regime": regime, "catalyst": catalyst, "study_id": s})
        mu, sd = gp_predict(final[name], make_features(frame).values, return_std=True)
        mus.append(mu); sds.append(sd)
    mus, sds = np.array(mus), np.array(sds)
    return mus.mean(0), np.sqrt((sds ** 2).mean(0) + mus.var(0))

cases = [c for c in [("continuous", "none"), ("batch (slow)", "none"), ("batch (slow)", "basic"), ("batch (slow)", "acidic")]
         if c[0] in REGIMES and c[1] in CATALYSTS]
tgrid = np.linspace(350, 650, 61)
fig, axes = plt.subplots(len(cases), 4, figsize=(7.2, 1.8 * len(cases) + 0.3), sharex=True, squeeze=False)
for i, (reg, cat) in enumerate(cases):
    for j, name in enumerate(NAMES):
        ax = axes[i, j]
        mu, sd = predict_new(tgrid, reg, cat, name)
        ax.fill_between(tgrid, np.clip(mu - 1.96 * sd, 0, 100), np.clip(mu + 1.96 * sd, 0, 100), color=COLORS[0], alpha=0.15,
                        label="95% interval")
        ax.plot(tgrid, np.clip(mu, 0, 100), color=COLORS[0], lw=1.5, label="Mean prediction")
        obs = df[(df.regime == reg) & (df.catalyst == cat)]
        ax.scatter(obs["T"], obs[TARGETS[name]], s=5, color=GREY, zorder=0, label="Data (this regime/catalyst)")
        if i == 0: ax.set_title(name)
        if i == len(cases) - 1: ax.set_xlabel("Temperature (°C)")
        if j == 0: ax.set_ylabel(f"{reg}, {cat}\nyield (wt%)", fontsize=7.5)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.03))
save(fig, 8, "Plot 8 - Yield-temperature curves for a new process (bands: 95% interval incl. between-study spread)")


#%% Plot 9 - Mass-balance consistency across the operating window
grid = []
for reg in REGIMES:
    for cat in CATALYSTS:
        row = {"regime": reg, "catalyst": cat}
        for name in NAMES:
            row[name] = np.clip(predict_new(tgrid, reg, cat, name)[0], 0, 100)
        grid.append(pd.DataFrame({"T": tgrid, **row}))
grid = pd.concat(grid, ignore_index=True)
grid["Styrene + BTX"] = grid["Styrene"] + grid["BTX"]
v1 = (grid["Styrene + BTX"] > grid["Total aromatics"] + 0.5).mean()
v2 = (grid["Total aromatics"] > grid["Liquid"] + 0.5).mean()
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6))
for ax, (xc, yc, v) in zip(axes, [("Styrene + BTX", "Total aromatics", v1), ("Total aromatics", "Liquid", v2)]):
    for k, reg in enumerate(REGIMES):
        g = grid[grid.regime == reg]
        ax.scatter(g[xc], g[yc], s=4, color=COLORS[k % 4], alpha=0.6, label=reg)
    lim = [0, 105]; ax.plot(lim, lim, "--", color=GREY, lw=1)
    ax.fill_between(lim, [0, 0], lim, color="#eb6834", alpha=0.06)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"Predicted {xc} (wt%)"); ax.set_ylabel(f"Predicted {yc} (wt%)")
    ax.set_title(f"{yc} ≥ {xc} required\nviolations: {100 * v:.1f}% of grid points")
axes[0].legend(frameon=False, title="Regime", markerscale=2, title_fontsize=7.5)
save(fig, 9, "Plot 9 - Mass-balance check over 350-650 °C, all regime x catalyst combinations (shaded = infeasible side)")


#%% Plot 10 - Range-of-validity map: where the training data lie
combos = (df.dropna(subset=["T"]).groupby(["regime", "catalyst"]).size().sort_values().index.tolist())
fig, ax = plt.subplots(figsize=(6.8, 0.33 * len(combos) + 0.9))
for k, (reg, cat) in enumerate(combos):
    g = df[(df.regime == reg) & (df.catalyst == cat)].dropna(subset=["T"])
    ax.plot([g["T"].min(), g["T"].max()], [k, k], color=COLORS[0], lw=6, alpha=0.25, solid_capstyle="round")
    ax.scatter(g["T"], np.full(len(g), k), s=10, color=COLORS[0], edgecolor="white", linewidth=0.4, zorder=3)
    ax.text(g["T"].max() + 8, k, f"{len(g)} runs, {g['study_id'].nunique()} stud{'y' if g['study_id'].nunique() == 1 else 'ies'}", va="center", fontsize=7.5)
ax.axvline(500, color=COLORS[1], lw=1.5, ls="--"); ax.text(503, len(combos) - 0.4, "design point 500 °C", color=COLORS[1], fontsize=7.5)
ax.set_yticks(range(len(combos))); ax.set_yticklabels([f"{r} | {c}" for r, c in combos])
ax.set_xlabel("Temperature (°C)"); ax.set_xlim(300, 880); ax.grid(axis="y", alpha=0)
save(fig, 10, "Plot 10 - Range of validity: the model interpolates inside the shaded spans; outside them it extrapolates")


#%% Plot 11 - GPR predictions vs the yields used in the process model (Aspen stream table, Fig. SI6)
# Stream 8 (pyrolysis product, 12,395.83 kg/h) from 10,416.67 kg/h PS. Component wt% from the SI stream table.
# Assumptions (edit if your stream mapping differs): C8H10 counted as xylene (BTX); C9H12 counted as other
# aromatics; C10H22 (decane) counted as liquid but not aromatic. Design point 500 °C, continuous, ZnO = basic.
PS_FEED, S8_FLOW = 10416.67, 12395.83
s8 = {"C6H6": 1.95, "C7H8": 1.98, "C8H10": 1.98, "C8H8": 45.61, "C9H12": 19.53, "C10H22": 9.69}
to_feed = lambda wt: S8_FLOW * wt / 100 / PS_FEED * 100
aspen = {"Styrene": to_feed(s8["C8H8"]),
         "BTX": to_feed(s8["C6H6"] + s8["C7H8"] + s8["C8H10"]),
         "Total aromatics": to_feed(s8["C8H8"] + s8["C6H6"] + s8["C7H8"] + s8["C8H10"] + s8["C9H12"]),
         "Liquid": to_feed(sum(s8.values()))}
DESIGN_T = 500
fig, ax = plt.subplots(figsize=(5.6, 3.6))
x = np.arange(len(NAMES)); w = 0.26
pcases = [("continuous", "basic"), ("continuous", "none")]
pcases = [c for c in pcases if c[0] in REGIMES and c[1] in CATALYSTS]
ax.bar(x - w, [aspen[n] for n in NAMES], w, color="#c9c9c9", label="Process model (stream 8)")
for k, (reg, cat) in enumerate(pcases):
    mu = np.array([predict_new([DESIGN_T], reg, cat, n)[0][0] for n in NAMES])
    sd = np.array([predict_new([DESIGN_T], reg, cat, n)[1][0] for n in NAMES])
    lo, hi = np.clip(mu - 1.96 * sd, 0, 100), np.clip(mu + 1.96 * sd, 0, 100)
    ax.bar(x + k * w, mu, w, color=COLORS[k], yerr=[mu - lo, hi - mu], ecolor="#444444", capsize=3,
           label=f"GPR {DESIGN_T} °C, {reg}, catalyst: {cat} (95% interval)")
for xi, n in zip(x, NAMES):
    ax.text(xi - w, aspen[n] + 1, f"{aspen[n]:.1f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("Yield (wt% of PS feed)"); ax.set_ylim(0, 105)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
save(fig, 11, "Plot 11 - Process-model yields vs GPR prediction at the design point")
print("Process-model yields (wt% of feed):", {k: round(v, 1) for k, v in aspen.items()})

#%% Save the numbers behind the plots
with pd.ExcelWriter(os.path.join(HERE, "PS_GPR_diagnostics_data.xlsx")) as xw:
    cmp.round(3).to_excel(xw, sheet_name="5 Model comparison", index=False)
    lc.groupby(["Output", "fraction"]).mean(numeric_only=True).round(3).reset_index().to_excel(xw, sheet_name="3 Learning curve", index=False)
    imp.round(3).to_excel(xw, sheet_name="6 Importance", index=False)
    grid.round(2).to_excel(xw, sheet_name="9 Mass-balance grid", index=False)
    pd.DataFrame({"Output": NAMES, "Process model (wt% of feed)": [aspen[n] for n in NAMES]}).round(2).to_excel(
        xw, sheet_name="11 Process model", index=False)
print("Saved fig_diag_01 ... fig_diag_11 and PS_GPR_diagnostics_data.xlsx to", HERE)
