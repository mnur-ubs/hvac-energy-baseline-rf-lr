"""Hybrid reanalysis: manuscript configuration as PRIMARY analysis + r2 variants as SENSITIVITY.

PRIMARY (matches the current manuscript, no hyper-parameter tuning)
  features  one-hot Hour (ref 00) + one-hot DoW (ref Monday) + Lag-1 + Lag-24
            (one-hot columns in the round-1 order, see note in the report)
  models    LR = OLS; RF = scikit-learn defaults, n_estimators=100, random_state=42
  selection LR vs RF by validation RMSE only -> refit on Train+Val -> score once on Test
  ablations without Lag-1, calendar-only (LR and RF)
SENSITIVITY (reported separately, never used for selection)
  (a) RF tuned on validation (r2 grid), one-hot   (b) cyclic encoding, LR + default RF
  (c) tuned RF, cyclic (= r2 selected model)        (d) default RF, one-hot in numeric column order

Shared code (data loading with rounded hour, split, folds, metrics, DM-HAC, G14) lives in reanalysis_common.py.
Outputs: reanalysis_hybrid.md and reanalysis_hybrid/ (CSVs + figures).

Usage:  python reanalysis_hybrid.py ["Perhitungan LR vs RF.xlsx"]
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import reanalysis_common as rc
from reanalysis_common import SEED, LAGSETS, MET, BENCH, RF_GRID, COL, group_of, metrics, dm_test, ci95

rc.setup()
HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Perhitungan LR vs RF.xlsx")
OUT = os.path.join(HERE, "reanalysis_hybrid"); FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
REPORT = os.path.join(HERE, "reanalysis_hybrid.md")
NW = 24                                   # Newey–West lags for DM-HAC
R = rc.Report(); md, table = R.md, R.table

# =============================================================================== DATA & SPLIT
D = rc.load_data(XLSX); df = D["df"]; n = len(df)
TR, VA, TE, TRVA = rc.split_idx(n)
assert (len(TR), len(VA), len(TE)) == (5241, 1747, 1748)
y = df.y.values; yva, yte = y[VA], y[TE]
ts_te = df.ts.iloc[TE].reset_index(drop=True)

CAL = {"onehot": rc.calendar(df, "onehot", column_order="lexicographic"),   # = round-1 / manuscript column order
       "onehot_num": rc.calendar(df, "onehot", column_order="numeric"),
       "cyclic": rc.calendar(df, "cyclic")}
def X_of(enc, lags): return pd.concat([CAL[enc], df[LAGSETS[lags]]], axis=1)

def lr(): return LinearRegression()
def rf_default(): return RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
def rf_tuned(p): return lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **p)

SPECS = {}
def add(name, kind, enc, lags, est, group, label, match_lr=None, params="default (100 trees)" ):
    SPECS[name] = dict(kind=kind, enc=enc, lags=lags, est=est, group=group, label=label, match_lr=match_lr,
                       params="OLS" if kind == "LR" else params)

LAG_TXT = {"all": "Hour + DoW + Lag-1 + Lag-24", "noLag1": "Hour + DoW + Lag-24", "calendar": "Hour + DoW only"}
for m in ("LR", "RF"):
    for l in LAGSETS:
        add(m if l == "all" else f"{m}_{l}", m, "onehot", l, lr if m == "LR" else rf_default, "primary", f"{m}: {LAG_TXT[l]}")
PRIMARY = ["LR", "LR_noLag1", "LR_calendar", "RF", "RF_noLag1", "RF_calendar"]
ORDER = BENCH + PRIMARY

# --- sensitivity: tune RF on validation (same grid / tree count as r2)
def tune(enc):
    X = X_of(enc, "all"); rows = []
    for p in RF_GRID:
        pv = rf_tuned(p)().fit(X.iloc[TR], y[TR]).predict(X.iloc[VA])
        rows.append(dict(encoding=enc, min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"], **metrics(yva, pv)))
    g = pd.DataFrame(rows); b = g.loc[g.RMSE.idxmin()]
    return dict(min_samples_leaf=int(b.min_samples_leaf), max_features=float(b.max_features)), g
p_oh, grid_oh = tune("onehot"); p_cy, grid_cy = tune("cyclic")
ptxt = lambda p: f"tuned: min_samples_leaf={p['min_samples_leaf']}, max_features={p['max_features']}, 300 trees"
add("RF_tuned_onehot", "RF", "onehot", "all", rf_tuned(p_oh), "sens", "(a) RF tuned, one-hot", "LR", ptxt(p_oh))
add("LR_cyclic", "LR", "cyclic", "all", lr, "sens", "(b) LR, cyclic")
add("RF_cyclic", "RF", "cyclic", "all", rf_default, "sens", "(b) RF default, cyclic", "LR_cyclic")
add("RF_tuned_cyclic", "RF", "cyclic", "all", rf_tuned(p_cy), "sens", "(c) RF tuned, cyclic (= r2 selected)", "LR_cyclic", ptxt(p_cy))
add("RF_onehot_numeric", "RF", "onehot_num", "all", rf_default, "sens", "(d) RF default, one-hot numeric column order", "LR")
SENS = ["RF_tuned_onehot", "LR_cyclic", "RF_cyclic", "RF_tuned_cyclic", "RF_onehot_numeric"]
ALL = PRIMARY + SENS

def fit_predict(name, fit_idx, ev_idx):
    s = SPECS[name]; X = X_of(s["enc"], s["lags"])
    mdl = s["est"]().fit(X.iloc[fit_idx], y[fit_idx]); return mdl.predict(X.iloc[ev_idx]), mdl

# =============================================================================== VALIDATION / TEST / ROLLING
pva = {k: fit_predict(k, TR, VA)[0] for k in ALL}; pva.update(rc.bench_pred(df, TR, VA))
val = pd.DataFrame({k: metrics(yva, p) for k, p in pva.items()}).T
SELECTED = val.loc[["LR", "RF"], "RMSE"].idxmin()
OTHER = "LR" if SELECTED == "RF" else "RF"

pte, fitted = {}, {}
for k in ALL: pte[k], fitted[k] = fit_predict(k, TRVA, TE)
pte.update(rc.bench_pred(df, TRVA, TE))
test = pd.DataFrame({k: metrics(yte, p) for k, p in pte.items()}).T
for b in BENCH: test[f"Skill vs {b} (%)"] = 100 * (1 - test.RMSE / test.loc[b, "RMSE"])

fold_rows, oos = [], {k: np.full(n, np.nan) for k in ALL + BENCH}
for k, mo, fit_idx, ev_idx in rc.monthly_folds(df):
    preds = rc.bench_pred(df, fit_idx, ev_idx)
    for name in ALL: preds[name] = fit_predict(name, fit_idx, ev_idx)[0]
    for name, p in preds.items():
        oos[name][ev_idx] = p
        fold_rows.append(dict(model=name, fold=k, month=str(mo), n_train=len(fit_idx), n_test=len(ev_idx), **metrics(y[ev_idx], p)))
folds = pd.DataFrame(fold_rows)
OOS = ~np.isnan(oos["Mean"]); yo = y[OOS]
pooled = pd.DataFrame({k: metrics(yo, oos[k][OOS]) for k in ALL + BENCH}).T
for b in BENCH: pooled[f"Skill vs {b} (%)"] = 100 * (1 - pooled.RMSE / pooled.loc[b, "RMSE"])
piv = lambda col: folds.pivot(index="model", columns="month", values=col)

def dm_row(a, b):
    """DM-HAC(24) on test and on pooled folds; d = e²_a − e²_b (positive ⇒ b more accurate)."""
    t, p = dm_test(yte, pte[a], pte[b], NW), dm_test(yo, oos[a][OOS], oos[b][OOS], NW)
    return {"Test mean d": t["mean_d"], "Test DM-HAC": t["DM"], "p (test)": t["p"],
            "Pooled mean d": p["mean_d"], "Pooled DM-HAC": p["DM"], "p (pooled)": p["p"]}

def fold_tests(a, b):
    fa = piv("RMSE").loc[a]; fb = piv("RMSE").loc[b]; d = fa - fb
    return {"Fold ΔRMSE (a−b) mean": d.mean(), "Wilcoxon p": stats.wilcoxon(fa, fb).pvalue,
            "Folds b better": f"{int((d > 0).sum())}/{len(d)}"}

# =============================================================================== DEVIATION
E_tot = yte.sum()
def deviation(name):
    r = yte - pte[name]; s_val, s_te = val.loc[name, "RMSE"], test.loc[name, "RMSE"]; tau = 2 * s_val
    big = r > tau; ipos = int(np.argmax(r))
    return {"Model": name, "PRF: share of hours Y>Ŷ (%)": (r > 0).mean() * 100,
            "E⁺ (kWh)": np.clip(r, 0, None).sum(), "E⁺ % of use": np.clip(r, 0, None).sum() / E_tot * 100,
            "E⁻ (kWh)": np.clip(-r, 0, None).sum(), "Net Σ(Y−Ŷ) (kWh)": r.sum(), "Net % of use": r.sum() / E_tot * 100,
            "τ = 2·RMSE_val (kW)": tau, "Hours e > τ": int(big.sum()), "Σ e over hours > τ (kWh)": r[big].sum(),
            "…% of use": r[big].sum() / E_tot * 100, "Σ (e − τ) over hours > τ (kWh)": (r[big] - tau).sum(),
            "Max positive e (kW)": r[ipos], "at": str(ts_te[ipos]), "Max/σ_val": r[ipos] / s_val, "Max/σ_test": r[ipos] / s_te}
dev = pd.DataFrame([deviation(SELECTED), deviation(OTHER)])

# =============================================================================== CONSISTENCY WITH MANUSCRIPT
dm_classic = dm_test(yte, pte["LR"], pte["RF"], 0); dm_hac = dm_test(yte, pte["LR"], pte["RF"], NW)
dRF = deviation("RF")
CHECK = [  # (item, manuscript value as printed, this run, decimals)
    ("RF validation RMSE", "2.340", val.loc["RF", "RMSE"]), ("LR validation RMSE", "2.668", val.loc["LR", "RMSE"]),
    ("RF test MAE", "2.158", test.loc["RF", "MAE"]), ("RF test RMSE", "2.976", test.loc["RF", "RMSE"]),
    ("RF test R²", "0.003", test.loc["RF", "R2"]),
    ("LR test MAE", "2.152", test.loc["LR", "MAE"]), ("LR test RMSE", "2.796", test.loc["LR", "RMSE"]),
    ("LR test R²", "0.120", test.loc["LR", "R2"]),
    ("DM statistic LR vs RF (test)", "-2.85", dm_classic["DM"]), ("DM p-value (test)", "0.004", dm_classic["p"]),
    ("Calendar-only RF test RMSE", "2.775", test.loc["RF_calendar", "RMSE"]),
    ("RF hours with e > τ (test)", "120", dRF["Hours e > τ"]), ("RF kWh over hours > τ", "821", dRF["Σ e over hours > τ (kWh)"]),
    ("RF kWh > τ as % of use", "16.3", dRF["…% of use"]),
]
cons = []
for item, ms, v in CHECK:
    dp = len(ms.split(".")[1]) if "." in ms else 0
    ok = abs(round(float(v), dp) - float(ms)) < 1e-9
    cons.append({"Item": item, "Manuscript": ms, "This run": v if isinstance(v, (int, np.integer)) else float(v),
                 "Matches at printed precision": ok})
cons_vals = [r["This run"] for r in cons]; cons = pd.DataFrame(cons); cons["This run"] = pd.Series(cons_vals, dtype=object)

# =============================================================================== REPORT
md("# Hybrid reanalysis — manuscript configuration (primary) with sensitivity analyses")
md()
md(f"Generated by `reanalysis_hybrid.py` (seed = {SEED} everywhere). Data loading (timestamps rounded to the hour), the 60/20/20 "
   "split, the monthly folds, metrics, DM-HAC and G14 come from `reanalysis_common.py`, shared with `reanalysis_r2.py`. "
   "All numbers have 3 decimals. Residual e = Y − Ŷ (positive = actual above baseline). CV(RMSE) and NMBE in %.")
md()
md("**Primary configuration (manuscript, no tuning):** one-hot Hour (ref 00) + one-hot DoW (ref Monday) + Lag-1 + Lag-24; "
   "LR = OLS; RF = scikit-learn defaults with n_estimators = 100, random_state = 42. LR vs RF is decided by validation RMSE only; "
   "the chosen model is refitted on Train+Validation and scored once on Test. Benchmarks: constant mean (mean of the fitting "
   "window, i.e. Train for validation and Train+Validation for test) and persistence Ŷ(t) = Y(t−1).")
md()
md("*Note on column order.* The one-hot columns are kept in the order of the round-1 script (`pd.get_dummies` on string "
   "labels: hour_1, hour_10 … hour_19, hour_2, hour_20 … hour_9, dow_1 … dow_6). With `max_features = 1.0` the RF breaks ties between "
   "equally good splits by column position, so this order is needed to reproduce the manuscript to the last digit; the encoding "
   "itself is identical. Sensitivity (d) shows the size of this effect.")
md()

md("## 1. Data and split")
md()
md(f"{D['n_raw']:,} raw hourly rows → **{n:,}** rows after Lag-24 (first 24 h dropped).")
md()
table(pd.DataFrame([(nm, len(ix), str(df.ts.iloc[ix[0]]), str(df.ts.iloc[ix[-1]]), y[ix].mean(), y[ix].std())
                    for nm, ix in [("Train", TR), ("Validation", VA), ("Test (untouched)", TE)]],
                   columns=["Split", "Rows", "Start", "End", "Mean Y (kW)", "SD Y (kW)"]), index=False)

md("## 2. Consistency check against the manuscript")
md()
table(cons, index=False)
n_ok = int(cons["Matches at printed precision"].sum())
md(f"**{n_ok} of {len(cons)} values reproduce exactly** at the precision printed in the manuscript.")
md()
md(f"- The manuscript's DM value is the *classic* Diebold–Mariano statistic (h = 1, no autocorrelation correction): "
   f"DM = {dm_classic['DM']:.3f}, p = {dm_classic['p']:.3f}. With the Newey–West long-run variance (24 lags, HLN correction) "
   f"used throughout this report it becomes DM-HAC = {dm_hac['DM']:.3f}, p = {dm_hac['p']:.3f}. "
   f"The sign convention is d = e²_LR − e²_RF, so a **negative** statistic means **LR is more accurate than RF on the test set**. "
   "Hourly loss differentials are autocorrelated, so the HAC version should be cited.")
md("- Rolling-origin results in this report use the 9 monthly expanding-window folds (Apr–Dec 2025) of `reanalysis_r2.py`. "
   "The round-1 `reanalysis.py` used 5 consecutive 720-hour folds ending 31 Dec, so any rolling-origin mean ± SD quoted "
   "from round 1 will not match the numbers in Section 5.")
mism = cons[~cons["Matches at printed precision"]]
if len(mism):
    md("- Values that do not match: " + "; ".join(f"{r.Item} (manuscript {r.Manuscript}, this run {r._3:.3f})" for r in mism.itertuples()) + ".")
md()

md("## 3. Validation and model selection")
md()
md("Fitted on Train, scored on Validation. Only LR and RF (all features) are candidates; ablations and benchmarks are shown for context.")
md()
VCOL = MET + ["G14"]
table(val.loc[ORDER, VCOL].rename_axis("Model"))
md(f"**Selected: {SELECTED}** (validation RMSE {val.loc[SELECTED, 'RMSE']:.3f} vs {val.loc[OTHER, 'RMSE']:.3f} kW for {OTHER}). "
   "This choice was locked before the test set was scored.")
md()

md("## 4. Untouched test set (refit on Train+Validation, scored once)")
md()
md(f"Test window {ts_te.iloc[0]} → {ts_te.iloc[-1]} (n = {len(TE):,}). Skill = 1 − RMSE_model / RMSE_benchmark (%).")
md()
TCOL = MET + ["G14", "Skill vs Mean (%)", "Skill vs Persistence (%)"]
table(test.loc[ORDER, TCOL].rename_axis("Model"))
md(f"Selected model ({SELECTED}) on test: RMSE {test.loc[SELECTED, 'RMSE']:.3f} kW, R² {test.loc[SELECTED, 'R2']:.3f}; "
   f"{OTHER}: RMSE {test.loc[OTHER, 'RMSE']:.3f} kW, R² {test.loc[OTHER, 'R2']:.3f}.")
md()

md("## 5. Rolling-origin evaluation (9 monthly expanding-window folds)")
md()
md("Initial window 2025-01-02 → 2025-03-31; each fold refits every model on all data before the month and scores that month. "
   "Cells: mean ± SD [95 % CI, t with 8 df] across folds.")
md()
table(folds[folds.model == "Mean"][["fold", "month", "n_train", "n_test"]], index=False)
roll = []
for k in ORDER:
    g = folds[folds.model == k]; r = {"Model": k}
    for mt in MET:
        mu, sd, lo, hi = ci95(g[mt]); r[mt] = f"{mu:.3f} ± {sd:.3f} [{lo:.3f}, {hi:.3f}]"
    roll.append(r)
roll = pd.DataFrame(roll); table(roll, index=False)
md("**RMSE (kW) per fold**")
md()
table(piv("RMSE").loc[ORDER].rename_axis("Model"))
md(f"**Pooled out-of-sample metrics over all folds** (n = {OOS.sum():,} h) and skill vs benchmarks")
md()
table(pooled.loc[ORDER, TCOL].rename_axis("Model"))

md("## 6. Significance tests")
md()
md(f"Diebold–Mariano, squared-error loss, Newey–West long-run variance with {NW} lags, HLN small-sample factor, p from t(T−1). "
   "d = e²_A − e²_B, so **positive d ⇒ B is more accurate**. 'Pooled' = the concatenated out-of-sample predictions of the 9 folds. "
   "Wilcoxon signed-rank test on the 9 fold RMSEs (A vs B).")
md()
md("**RF vs LR (primary)**")
md()
rf_lr = {"A": "LR", "B": "RF", "Test DM classic (h=1)": dm_classic["DM"], "p classic": dm_classic["p"], **dm_row("LR", "RF"),
         **fold_tests("LR", "RF")}
table(pd.DataFrame([rf_lr]).rename(columns={"Folds b better": "Folds RF better"}), index=False)
md("**Each model vs the benchmarks**")
md()
bm = []
for k in PRIMARY:
    for b in BENCH:
        bm.append({"A (benchmark)": b, "B (model)": k, **dm_row(b, k), "Folds model better": fold_tests(b, k)["Folds b better"]})
bm = pd.DataFrame(bm); table(bm, index=False)
md("**Lag ablation** (A = reduced model, B = full model; positive d ⇒ the full model is more accurate)")
md()
abl = []
for m in ("RF", "LR"):
    for l in ("noLag1", "calendar"):
        a = f"{m}_{l}"; abl.append({"A (reduced)": a, "B (full)": m, **dm_row(a, m), "Folds full better": fold_tests(a, m)["Folds b better"]})
abl = pd.DataFrame(abl); table(abl, index=False)

md("## 7. ASHRAE Guideline 14 (hourly: CV(RMSE) ≤ 30 %, |NMBE| ≤ 10 %)")
md()
md("CV(RMSE) = RMSE / Ȳ × 100; NMBE = Σ(Y − Ŷ) / (n · Ȳ) × 100; out-of-sample, so no degrees-of-freedom correction.")
md()
g14 = pd.DataFrame({"Val CV(RMSE)": val.CVRMSE, "Val NMBE": val.NMBE, "Val pass": val.G14,
                    "Test CV(RMSE)": test.CVRMSE, "Test NMBE": test.NMBE, "Test pass": test.G14,
                    "Folds passed": folds.groupby("model").G14.sum().astype(int)}).loc[ORDER]
table(g14.rename_axis("Model"))
md("**CV(RMSE) (%) per fold**")
md()
table(piv("CVRMSE").loc[ORDER].rename_axis("Model"))
md("**NMBE (%) per fold**")
md()
table(piv("NMBE").loc[ORDER].rename_axis("Model"))
md(f"Models passing G14 — validation: {int(val.loc[ORDER].G14.sum())}/{len(ORDER)}, test: {int(test.loc[ORDER].G14.sum())}/{len(ORDER)}, "
   f"folds: {int(folds[folds.model.isin(ORDER)].G14.sum())}/{len(ORDER) * len(rc.MONTHS)}.")
md()

md("## 8. LR (OLS) coefficients, primary model fitted on Train+Validation")
md()
Xl = sm.add_constant(X_of("onehot", "all"))
ols = sm.OLS(y[TRVA], Xl.iloc[TRVA]).fit(cov_type="HAC", cov_kwds={"maxlags": NW})
assert np.allclose(ols.params.values[1:], fitted["LR"].coef_)
coef = pd.DataFrame({"coef": ols.params, "HAC SE": ols.bse, "z": ols.tvalues, "p": ols.pvalues,
                     "CI low": ols.conf_int()[0], "CI high": ols.conf_int()[1]})
coef = coef.loc[["const"] + [f"hour_{h}" for h in range(1, 24)] + [f"dow_{d}" for d in range(1, 7)] + ["lag1", "lag24"]].rename_axis("term")
md(f"Newey–West HAC standard errors ({NW} lags); in-sample R² = {ols.rsquared:.3f}, n = {int(ols.nobs):,}.")
md()
table(coef)

md("## 9. RF feature importance (primary RF models, fitted on Train+Validation)")
md()
md("MDI = impurity importance summed over the columns of each group (%). Permutation = increase in test MSE (kW²) when the whole "
   "group is permuted jointly, mean ± SD over 10 repeats (seed 42).")
md()
GROUPS = ["Hour", "DayOfWeek", "Lag-1", "Lag-24"]
imp_rows = []
for k in ("RF", "RF_noLag1", "RF_calendar"):
    s = SPECS[k]; X = X_of(s["enc"], s["lags"]); groups = [group_of(c) for c in X.columns]
    mdi = pd.Series(fitted[k].feature_importances_, index=X.columns).groupby(groups).sum() * 100
    perm = rc.grouped_permutation(fitted[k], X.iloc[TE], yte, groups, n_repeats=10, seed=SEED)
    for g in GROUPS:
        if g in mdi.index:
            imp_rows.append(dict(model=k, group=g, MDI=mdi[g], perm=perm[g][0], perm_sd=perm[g][1]))
imp = pd.DataFrame(imp_rows)
imp_tbl = imp.assign(Perm=lambda d: [f"{a:.3f} ± {b:.3f}" for a, b in zip(d.perm, d.perm_sd)])
imp_w = pd.concat({"MDI %": imp.pivot(index="model", columns="group", values="MDI"),
                   "Perm ΔMSE": imp_tbl.pivot(index="model", columns="group", values="Perm")}, axis=1)
imp_w = imp_w.reindex(columns=pd.MultiIndex.from_product([["MDI %", "Perm ΔMSE"], GROUPS])).loc[["RF", "RF_noLag1", "RF_calendar"]]
imp_w.columns = [f"{a}: {b}" for a, b in imp_w.columns]
table(imp_w.astype(object).fillna("—").rename_axis("Model"))

md(f"## 10. Deviation analysis on the untouched test set (selected model: {SELECTED})")
md()
md("PRF = share of hours with a positive residual (Y > Ŷ; corrected Eq. 9, a sign statistic, not waste). E⁺ = Σ max(Y − Ŷ, 0) "
   "kWh, E⁻ = Σ max(Ŷ − Y, 0) kWh, net = E⁺ − E⁻ (Δt = 1 h). τ = 2 × the model's validation RMSE (fixed before test). "
   f"'Σ e over hours > τ' is the round-1 definition (whole residual of the flagged hours); 'Σ (e − τ)' counts only the part above τ. "
   f"The {OTHER} row is shown for comparison. Total test consumption = {E_tot:.3f} kWh.")
md()
table(dev.set_index("Model").T.rename_axis("Quantity"))
r_sel = yte - pte[SELECTED]; s_val, s_te = val.loc[SELECTED, "RMSE"], test.loc[SELECTED, "RMSE"]
top = pd.DataFrame({"Timestamp": ts_te.astype(str), "Y (kW)": yte, "Ŷ (kW)": pte[SELECTED], "e (kW)": r_sel,
                    "e/σ_val": r_sel / s_val, "e/σ_test": r_sel / s_te})
top = top.reindex(top["e (kW)"].abs().sort_values(ascending=False).index).head(10)
md(f"**Top-10 |residuals| — {SELECTED}** (σ_val = {s_val:.3f} kW, σ_test = {s_te:.3f} kW)")
md()
table(top, index=False)
hr = rc.hourly_residual(df.hour.values[TE], r_sel)
hr_prf = pd.DataFrame({"h": df.hour.values[TE], "pos": r_sel > 0}).groupby("h").pos.mean() * 100
hr_out = pd.DataFrame({"Mean e (kW)": hr["mean"], "95% CI low": hr.lo, "95% CI high": hr.hi,
                       "CI excludes 0": hr["CI excludes 0"], "PRF (%)": hr_prf, "n": hr["count"].astype(int)}).rename_axis("Hour")
md(f"**Mean test residual by hour of day — {SELECTED}** (95 % CI = mean ± 1.96·SE)")
md()
table(hr_out)
md(f"Hours whose CI excludes 0: {[int(h) for h in hr.index[hr['CI excludes 0']]]}.")
md()

md("## 11. Test-window check")
md()
sub_rows = []
for nm, a, b in rc.SUBPERIODS:
    msk = np.ones(len(ts_te), bool)
    if a: msk &= (ts_te >= a).values
    if b: msk &= (ts_te < b).values
    r = {"Sub-period": nm, "n": int(msk.sum()), "Mean Y (kW)": yte[msk].mean()}
    for k in ORDER: r[f"RMSE {k}"] = float(np.sqrt(np.mean((yte[msk] - pte[k][msk]) ** 2)))
    sub_rows.append(r)
sub = pd.DataFrame(sub_rows)
md("**RMSE (kW) by sub-period inside the test window**")
md()
table(sub, index=False)
mon = df.assign(month=df.ts.dt.to_period("M").astype(str)).groupby("month").y
mon = pd.DataFrame({"mean": mon.mean(), "std": mon.std(), "max": mon.max(), "share of hours ≥ 8 kW (%)": mon.apply(lambda s: (s >= 8).mean() * 100)})
md("**Monthly load statistics (kW)**")
md()
table(mon.rename_axis("Month"))
sp = folds.assign(period=np.where(folds.month.isin(["2025-10", "2025-11", "2025-12"]), "Oct–Dec", "Apr–Sep"))
sp = sp[sp.model.isin(ORDER)].groupby(["model", "period"]).RMSE.mean().unstack().loc[ORDER]
sp.columns = [f"Mean fold RMSE {c}" for c in sp.columns]
md("**Rolling folds: Oct–Dec vs Apr–Sep** (mean of fold RMSEs, kW)")
md()
table(sp.rename_axis("Model"))

md("## 12. Sensitivity analyses (not used for selection)")
md()
md("Each variant is fitted exactly like the primary models (Train → Validation, Train+Validation → Test, same 9 folds). "
   "Tuned variants use the r2 grid (min_samples_leaf ∈ {1, 5, 20, 50} × max_features ∈ {1.0, 0.5}, 300 trees), selected by "
   "validation RMSE; the fold refits keep the selected hyper-parameters. DM-HAC compares each RF variant with the LR that uses the "
   "same encoding (d = e²_LR − e²_RF; positive ⇒ RF more accurate).")
md()
sens = []
for k in SENS:
    s = SPECS[k]; r = {"Variant": s["label"], "Model key": k, "Hyper-parameters": s["params"], "Val RMSE": val.loc[k, "RMSE"]}
    r.update({f"Test {m}": test.loc[k, m] for m in MET})
    r["Pooled rolling RMSE"] = pooled.loc[k, "RMSE"]
    if s["match_lr"]:
        a = s["match_lr"]; d = dm_row(a, k)
        r.update({"vs LR": a, "Test DM-HAC": d["Test DM-HAC"], "p (test)": d["p (test)"], "Pooled DM-HAC": d["Pooled DM-HAC"],
                  "p (pooled)": d["p (pooled)"], "LR pooled rolling RMSE": pooled.loc[a, "RMSE"]})
    sens.append(r)
sens = pd.DataFrame(sens)
table(sens.astype(object).fillna("—"), index=False)
r2_path = os.path.join(HERE, "reanalysis_r2", "test_metrics.csv")
if os.path.exists(r2_path):
    r2t = pd.read_csv(r2_path, index_col=0)
    if "RF / cyclic / all" in r2t.index:
        md(f"Cross-check with `reanalysis_r2.py`: (c) test RMSE {test.loc['RF_tuned_cyclic', 'RMSE']:.3f} kW here vs "
           f"{r2t.loc['RF / cyclic / all', 'RMSE']:.3f} kW in r2; LR cyclic {test.loc['LR_cyclic', 'RMSE']:.3f} vs "
           f"{r2t.loc['LR / cyclic / all', 'RMSE']:.3f} kW.")
        md()
        md(f"(a) differs slightly from r2's `RF / onehot / all` ({test.loc['RF_tuned_onehot', 'RMSE']:.3f} here vs "
           f"{r2t.loc['RF / onehot / all', 'RMSE']:.3f} kW) only because r2 orders the one-hot columns numerically.")
        md()
md("RF tuning grids (validation RMSE, fitted on Train):")
md()
grid = pd.concat([grid_oh, grid_cy]); table(grid[["encoding", "min_samples_leaf", "max_features", "MAE", "RMSE", "R2"]], index=False)

# =============================================================================== FIGURES
rc.apply_style(plt); figs = {}
show = ["LR", "RF"]
fig, ax = plt.subplots(2, 2, figsize=(10, 7.4), gridspec_kw={"height_ratios": [1.15, 1]})
w = slice(0, 14 * 24)
for j, k in enumerate(show):
    col = COL[k]; a = ax[0, j]; lim = [0, max(yte.max(), pte[k].max()) * 1.03]
    a.scatter(yte, pte[k], s=8, alpha=.35, color=col, edgecolors="none")
    a.plot(lim, lim, color="#52514e", lw=1, ls="--", label="y = x")
    sl = np.polyfit(yte, pte[k], 1); a.plot(lim, np.polyval(sl, lim), color=col, lw=2, label=f"fit, slope {sl[0]:.3f}")
    a.set(xlim=lim, ylim=lim, xlabel="Actual (kW)", ylabel="Predicted (kW)",
          title=f"{k}{'  [selected]' if k == SELECTED else ''} — test\nRMSE {test.loc[k, 'RMSE']:.3f}, R² {test.loc[k, 'R2']:.3f}")
    a.legend(fontsize=7, loc="upper left", frameon=False)
    b = ax[1, j]
    b.plot(ts_te[w], yte[w], color="#3d3c39", lw=1, label="Actual"); b.plot(ts_te[w], pte[k][w], color=col, lw=2, label="Predicted")
    b.set(ylabel="kW", title="First 14 days of test"); b.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    b.tick_params(axis="x", labelrotation=30, labelsize=7)
fig.tight_layout(); p = os.path.join(FIG, "hyb_pred_vs_actual.png"); fig.savefig(p); plt.close(fig); figs["Predicted vs actual (test)"] = p

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, k in zip(ax, [SELECTED, OTHER]):
    g = rc.hourly_residual(df.hour.values[TE], yte - pte[k]); col = COL[k]
    a.bar(g.index, g["mean"], width=0.75, color=[col if s else "#c3c2b7" for s in g["CI excludes 0"]])
    a.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="none", ecolor="#52514e", lw=1, capsize=2)
    a.axhline(0, color="#0b0b0b", lw=.8); a.set_xticks(range(0, 24, 2))
    a.set(title=f"{k}{' [selected]' if k == SELECTED else ''}: mean residual ± 95% CI (colored = CI excludes 0)",
          xlabel="Hour of day", ylabel="Mean residual Y − Ŷ (kW)")
fig.tight_layout(); p = os.path.join(FIG, "hyb_residuals_by_hour.png"); fig.savefig(p); plt.close(fig); figs["Residuals by hour (test)"] = p

lines = [("Mean", "Mean (benchmark)", COL["Mean"], "--"), ("Persistence", "Persistence (benchmark)", COL["Persistence"], ":"),
         ("LR", "LR", COL["LR"], "-"), ("RF", "RF", COL["RF"], "-"),
         ("RF_noLag1", "RF without Lag-1", COL["noLag1"], "-"), ("RF_calendar", "RF calendar-only", COL["calendar"], "-")]
mlab = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]; xs = range(len(mlab))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, colname, ttl in [(ax[0], "RMSE", "RMSE (kW) per fold"), (ax[1], "CVRMSE", "CV(RMSE) (%) per fold"), (ax[2], "NMBE", "NMBE (%) per fold")]:
    pv = piv(colname); a.axvspan(5.5, 8.5, color="#e9e8e4", zorder=0)
    for k, lbl, col, ls in lines:
        a.plot(xs, pv.loc[k].values, color=col, ls=ls, lw=2, marker="o", ms=4, label=lbl)
    a.set_xticks(xs, mlab); a.set(title=ttl, xlabel="Monthly fold (2025); shaded = Oct–Dec")
ax[1].axhline(30, color="#0b0b0b", lw=1, ls="--"); ax[1].text(0, 33, "G14 limit 30%", fontsize=7, color="#52514e")
ax[2].axhspan(-10, 10, color="#d9ead3", alpha=.5, zorder=0); ax[2].axhline(0, color="#0b0b0b", lw=.8)
ax[2].text(0, 11, "G14 band ±10%", fontsize=7, color="#52514e")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); p = os.path.join(FIG, "hyb_rolling_folds.png"); fig.savefig(p); plt.close(fig)
figs["Rolling-fold RMSE, CV(RMSE), NMBE"] = p

gcol = {"Hour": "#2a78d6", "DayOfWeek": "#eb6834", "Lag-1": "#1baf7a", "Lag-24": "#eda100"}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
models_i = ["RF", "RF_noLag1", "RF_calendar"]; yy = np.arange(3); hgt = 0.2
for a, val_col, xl in [(ax[0], "MDI", "MDI importance (%)"), (ax[1], "perm", "Grouped permutation, ΔMSE on test (kW²)")]:
    pv = imp.pivot(index="model", columns="group", values=val_col).reindex(index=models_i, columns=GROUPS)
    for k, g in enumerate(GROUPS):
        a.barh(yy + (k - 1.5) * hgt, np.nan_to_num(pv[g].values), height=hgt * 0.9, color=gcol[g], label=g)
    a.set_yticks(yy, ["RF (all)", "RF without Lag-1", "RF calendar-only"]); a.invert_yaxis(); a.axvline(0, color="#0b0b0b", lw=.8)
    a.set(xlabel=xl)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); p = os.path.join(FIG, "hyb_feature_importance.png"); fig.savefig(p); plt.close(fig)
figs["RF feature importance"] = p

md("## 13. Figures and files")
md()
for k, v in figs.items(): md(f"- {k}: `{os.path.relpath(v, HERE).replace(os.sep, '/')}`")
md()
files = {"validation_metrics": val, "test_metrics": test, "rolling_folds": folds, "rolling_pooled": pooled,
         "consistency_check": cons, "dm_rf_vs_lr": pd.DataFrame([rf_lr]), "dm_vs_benchmarks": bm, "dm_lag_ablation": abl,
         "ashrae_g14": g14, "lr_coefficients": coef, "rf_importance": imp, "deviation": dev, "top10_residuals": top,
         "hourly_residual": hr_out, "test_subperiods": sub, "monthly_load": mon, "sensitivity": sens, "rf_tuning_grid": grid}
for k, d in files.items(): d.to_csv(os.path.join(OUT, f"{k}.csv"), index=not isinstance(d.index, pd.RangeIndex))
pd.DataFrame({"ts": ts_te, "y": yte, **{k: pte[k] for k in ORDER + SENS}}).to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
json.dump(dict(selected=SELECTED, tuned_onehot=p_oh, tuned_cyclic=p_cy, split=[len(TR), len(VA), len(TE)]),
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)
md("CSV versions of every table: `reanalysis_hybrid/` (" + ", ".join(files) + ", test_predictions).")
# ------------------------------------------------------------------ key findings (inserted at the top)
sig = lambda p: "significant" if p < 0.05 else "not significant"
better = lambda d, a, b: b if d > 0 else a
dmp = dm_row("LR", "RF"); ft = fold_tests("LR", "RF"); nl = dm_row("RF_noLag1", "RF"); mb = dm_row("Mean", SELECTED)
best_te = test.loc[PRIMARY, "RMSE"].idxmin(); dsel = deviation(SELECTED)
KF = ["## Key findings", "",
 f"1. **Reproduction.** {n_ok} of {len(cons)} manuscript values are reproduced exactly by the primary configuration (Section 2).",
 f"2. **Selection vs test.** Validation selects **{SELECTED}** (RMSE {val.loc[SELECTED, 'RMSE']:.3f} vs {val.loc[OTHER, 'RMSE']:.3f}). "
 f"On the untouched test set {SELECTED} has RMSE {test.loc[SELECTED, 'RMSE']:.3f} and {OTHER} {test.loc[OTHER, 'RMSE']:.3f} kW. "
 f"Test DM-HAC: {better(dmp['Test mean d'], 'LR', 'RF')} more accurate, DM = {dmp['Test DM-HAC']:.3f}, p = {dmp['p (test)']:.3f} "
 f"({sig(dmp['p (test)'])}); the classic DM reported in the manuscript gives {dm_classic['DM']:.3f}, p = {dm_classic['p']:.3f}.",
 f"3. **Rolling folds.** Pooled RMSE LR {pooled.loc['LR', 'RMSE']:.3f} vs RF {pooled.loc['RF', 'RMSE']:.3f} kW; DM-HAC "
 f"{better(dmp['Pooled mean d'], 'LR', 'RF')} more accurate, DM = {dmp['Pooled DM-HAC']:.3f}, p = {dmp['p (pooled)']:.3f} "
 f"({sig(dmp['p (pooled)'])}); RF better in {ft['Folds b better']} folds, Wilcoxon p = {ft['Wilcoxon p']:.3f} ({sig(ft['Wilcoxon p'])}).",
 f"4. **Benchmarks.** {SELECTED} test skill: {test.loc[SELECTED, 'Skill vs Mean (%)']:.3f} % vs Mean (DM-HAC p = {mb['p (test)']:.3f}, "
 f"{sig(mb['p (test)'])}) and {test.loc[SELECTED, 'Skill vs Persistence (%)']:.3f} % vs Persistence; pooled folds "
 f"{pooled.loc[SELECTED, 'Skill vs Mean (%)']:.3f} % vs Mean (p = {mb['p (pooled)']:.3f}). "
 f"Lowest test RMSE among primary models: {best_te} ({test.loc[best_te, 'RMSE']:.3f} kW).",
 f"5. **Lag-1 ablation (RF).** Test RMSE with / without Lag-1 = {test.loc['RF', 'RMSE']:.3f} / {test.loc['RF_noLag1', 'RMSE']:.3f} kW "
 f"(DM-HAC p = {nl['p (test)']:.3f}, {sig(nl['p (test)'])}); pooled folds {pooled.loc['RF', 'RMSE']:.3f} / "
 f"{pooled.loc['RF_noLag1', 'RMSE']:.3f} kW (p = {nl['p (pooled)']:.3f}, {sig(nl['p (pooled)'])}).",
 f"6. **ASHRAE G14.** No primary model or benchmark passes on validation, test or any fold; lowest test CV(RMSE) "
 f"{test.loc[ORDER, 'CVRMSE'].min():.3f} % (limit 30 %)." if int(g14[['Val pass', 'Test pass']].values.sum()) + int(g14['Folds passed'].sum()) == 0
 else f"6. **ASHRAE G14.** See Section 7 for the models that pass.",
 f"7. **Deviation ({SELECTED}, test).** PRF {dsel['PRF: share of hours Y>Ŷ (%)']:.3f} %; E⁺ {dsel['E⁺ (kWh)']:.3f} kWh "
 f"({dsel['E⁺ % of use']:.3f} % of use) against E⁻ {dsel['E⁻ (kWh)']:.3f} kWh; net {dsel['Net Σ(Y−Ŷ) (kWh)']:.3f} kWh; "
 f"{dsel['Hours e > τ']} h above τ = {dsel['τ = 2·RMSE_val (kW)']:.3f} kW; max positive residual {dsel['Max positive e (kW)']:.3f} kW "
 f"at {dsel['at']} ({dsel['Max/σ_val']:.3f} σ_val).",
 f"8. **Sensitivity.** Across the RF sensitivity variants (a)–(d) the test RMSE ranges {sens[sens['Model key'].str.startswith('RF')]['Test RMSE'].min():.3f}–{sens[sens['Model key'].str.startswith('RF')]['Test RMSE'].max():.3f} kW; "
 "no RF variant is significantly better than its matching LR on the test set (DM-HAC), while all are on the pooled folds "
 "(Section 12)." if (sens.dropna(subset=['p (test)'])['p (test)'] >= 0.05).all() and (sens.dropna(subset=['p (pooled)'])['p (pooled)'] < 0.05).all()
 else "8. **Sensitivity.** See Section 12; conclusions differ across variants.", ""]
pos = R.lines.index("## 1. Data and split"); R.lines[pos:pos] = KF
R.save(REPORT)
print("\nSaved", REPORT)
