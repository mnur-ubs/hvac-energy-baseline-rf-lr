"""Analisis ulang LR vs RF untuk baseline konsumsi HVAC (lihat claude.md)."""
import sys
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
XLSX = "../Perhitungan LR vs RF.xlsx" if len(sys.argv) < 2 else sys.argv[1]

# 1. Data: hanya Timestamp & Active Power; fitur dibangun ulang dengan shift() yang benar
df = pd.read_excel(XLSX, sheet_name="Hasil Perhitungan Manual", usecols="A:B")
df.columns = ["ts", "y"]
df["ts"] = pd.to_datetime(df["ts"]); df["y"] = pd.to_numeric(df["y"])
raw_ts = df["ts"].copy()
df["ts"] = df["ts"].dt.round("h")                  # serial Excel ber-jitter (mis. 23:59:59.999 -> 00:00)
print("Timestamp yang dikoreksi pembulatan:", int((raw_ts != df.ts).sum()),
      "| di antaranya jam bergeser:", int((raw_ts.dt.hour != df.ts.dt.hour).sum()))
df = df.dropna().sort_values("ts").reset_index(drop=True)
print("Baris mentah:", len(df), "| rentang:", df.ts.min(), "->", df.ts.max(),
      "| duplikat:", int(df.ts.duplicated().sum()))
gaps = df.ts.diff().dropna()
print("Interval tidak 1 jam:", int((gaps != pd.Timedelta("1h")).sum()))
assert len(df) == 8760 and (gaps == pd.Timedelta("1h")).all()

df["hour"] = df.ts.dt.hour
df["dow"] = df.ts.dt.day_of_week                   # 0 = Senin
df["lag1"] = df.y.shift(1)                         # jam SEBELUMNYA
df["lag24"] = df.y.shift(24)                       # 24 jam SEBELUMNYA
df = df.dropna().reset_index(drop=True)
n = len(df); print("Baris setelah lag:", n)

# 2. Encoding kalender one-hot (sama untuk LR dan RF)
X_oh = pd.get_dummies(df[["hour", "dow"]].astype(str), drop_first=True).astype(float)

def feats(cols):
    return pd.concat([X_oh, df[cols]], axis=1)

def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    rmse = mean_squared_error(y, p) ** 0.5
    return dict(MAE=mean_absolute_error(y, p), RMSE=rmse, R2=r2_score(y, p),
                CVRMSE=rmse / y.mean() * 100, NMBE=(y - p).sum() / (len(y) * y.mean()) * 100)

def rf():
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)   # default scikit-learn, tanpa tuning

MODELS = {
    "LR":          (LinearRegression, ["lag1", "lag24"]),
    "RF":          (rf, ["lag1", "lag24"]),
    "RF_noLag1":   (rf, ["lag24"]),
    "RF_calendar": (rf, []),
}

def evaluate(fit_sl, eval_sl):
    """Latih tiap model pada fit_sl, evaluasi pada eval_sl. Kembalikan (tabel metrik, prediksi)."""
    res, pred = {}, {}
    y_ev = df.y.iloc[eval_sl]
    for name, (mk, cols) in MODELS.items():
        X = feats(cols); m = mk().fit(X.iloc[fit_sl], df.y.iloc[fit_sl])
        pred[name] = m.predict(X.iloc[eval_sl]); res[name] = metrics(y_ev, pred[name])
    pred["Mean"] = np.full(len(y_ev), df.y.iloc[fit_sl].mean())
    pred["Persistence"] = df.lag1.iloc[eval_sl].values
    res["Mean"] = metrics(y_ev, pred["Mean"]); res["Persistence"] = metrics(y_ev, pred["Persistence"])
    order = ["Mean", "Persistence", "LR", "RF", "RF_noLag1", "RF_calendar"]
    return pd.DataFrame(res).T.loc[order], pred

def dm_test(y, p_a, p_b, h=1):
    """Diebold–Mariano, loss kuadrat; d>0 berarti B lebih baik dari A. Varians HAC (Newey-West, lag h-1)
    dan koreksi Harvey-Leybourne-Newbold, p-value dari distribusi t(n-1)."""
    y = np.asarray(y, float)
    d = (y - p_a) ** 2 - (y - p_b) ** 2; T = len(d); dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dm, 2 * (1 - stats.norm.cdf(abs(dm))), hln, 2 * (1 - stats.t.cdf(abs(hln), T - 1))

def section(t):
    print("\n" + "=" * 90 + "\n" + t + "\n" + "=" * 90)

# ---------------------------------------------------------------- Pemeriksaan pipeline lama
section("A. PEMERIKSAAN: split lama (1.752 jam terakhir = test, tanpa validation)")
old_te = slice(n - 1752, n); old_tr = slice(0, n - 1752)
print("train:", n - 1752, "| test:", 1752, "| test mulai:", df.ts.iloc[n - 1752])
t_old, p_old = evaluate(old_tr, old_te); print(t_old.round(3))
print("(Naskah lama: RMSE RF ~2,850, LR ~2,919)")

# ---------------------------------------------------------------- Desain baru 60/20/20
section("B. SPLIT KRONOLOGIS 60/20/20")
i_tr, i_va = int(.6 * n), int(.8 * n)
tr, va, te = slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)
print(f"train={i_tr} | validation={i_va - i_tr} | test={n - i_va} | total={n}")
for nm, s in [("train", tr), ("validation", va), ("test", te)]:
    print(f"  {nm:<10} {df.ts.iloc[s].iloc[0]} -> {df.ts.iloc[s].iloc[-1]}")

section("C. VALIDATION (dasar pemilihan model; latih pada train)")
t_va, p_va = evaluate(tr, va); print(t_va.round(3))
cand = t_va.loc[["LR", "RF"]]
chosen = cand.RMSE.idxmin(); print("Model terpilih berdasarkan RMSE validation:", chosen)
rmse_va_rf = t_va.loc["RF", "RMSE"]

section("D. TEST (sekali, latih ulang pada train+validation)")
trva = slice(0, i_va)
t_te, p_te = evaluate(trva, te); print(t_te.round(3))
yte = df.y.iloc[te].values
print("\nPerbaikan RMSE relatif terhadap Mean (%):")
print((100 * (1 - t_te.RMSE / t_te.loc["Mean", "RMSE"])).round(2).to_string())
print("\nASHRAE G14 (hourly): CV(RMSE) <= 30% dan |NMBE| <= 10%")
g14 = pd.DataFrame({"CVRMSE": t_te.CVRMSE.round(2), "NMBE": t_te.NMBE.round(2),
                    "lolos": (t_te.CVRMSE <= 30) & (t_te.NMBE.abs() <= 10)}); print(g14)

section("E. UJI DIEBOLD–MARIANO pada test (loss kuadrat, h=1)")
for a, b in [("LR", "RF"), ("Persistence", "RF"), ("Persistence", "LR"), ("RF_noLag1", "RF")]:
    dm, p, hln, ph = dm_test(yte, p_te[a], p_te[b])
    print(f"{a:>11} vs {b:<3}: DM={dm:7.3f} p={p:.4f} | HLN={hln:7.3f} p={ph:.4f}   (DM>0 => {b} lebih baik)")
dm, p, _, _ = dm_test(df.y.iloc[old_te].values, p_old["LR"], p_old["RF"])
print(f"Split lama LR vs RF: DM={dm:.3f} p={p:.4f}")

section("F. ROLLING-ORIGIN (expanding window), 5 fold x 720 jam")
rows = []
for k in range(5):
    end_tr = n - (5 - k) * 720; sl = slice(end_tr, end_tr + 720)
    t_f, _ = evaluate(slice(0, end_tr), sl)
    for name, r in t_f.iterrows():
        rows.append(dict(r, model=name, fold=k + 1, start=df.ts.iloc[end_tr].date(),
                         end=df.ts.iloc[end_tr + 719].date()))
cv = pd.DataFrame(rows)
print(cv[["fold", "start", "end"]].drop_duplicates().to_string(index=False))
order = ["Mean", "Persistence", "LR", "RF", "RF_noLag1", "RF_calendar"]
agg = cv.groupby("model")[["MAE", "RMSE", "R2", "CVRMSE", "NMBE"]].agg(["mean", "std"]).loc[order]
print(agg.round(3))
print("\nRMSE per fold:"); print(cv.pivot(index="fold", columns="model", values="RMSE")[order].round(3))
print("\nR2 per fold:"); print(cv.pivot(index="fold", columns="model", values="R2")[order].round(3))

section("G. ANALISIS DEVIASI (RF, data test)")
resid = yte - p_te["RF"]
thr = 2 * rmse_va_rf
pos = np.clip(resid, 0, None)
print(f"Fraksi residual positif (Y > Ŷ): {(resid > 0).mean() * 100:.2f}%  ({(resid > 0).sum()} dari {len(resid)} jam)")
print(f"Total konsumsi test: {yte.sum():.1f} kWh")
print(f"Energi berlebih Σmax(Y-Ŷ,0): {pos.sum():.1f} kWh = {pos.sum() / yte.sum() * 100:.2f}% konsumsi")
print(f"Ambang 2xRMSE validation RF = {thr:.3f} kW: {(resid > thr).sum()} jam, "
      f"{resid[resid > thr].sum():.1f} kWh = {resid[resid > thr].sum() / yte.sum() * 100:.2f}% konsumsi")
ts_te = df.ts.iloc[te].reset_index(drop=True)
top = pd.DataFrame({"ts": ts_te, "y": yte, "yhat": p_te["RF"], "resid": resid}).nlargest(10, "resid")
print("\n10 deviasi positif terbesar:"); print(top.round(3).to_string(index=False))
full = df[["ts", "y"]].copy(); hit = full[(full.y - 8.890).abs() < 0.0005]
print("\nTimestamp dengan nilai y ≈ 8,890 kW (seluruh data):"); print(hit.to_string(index=False) if len(hit) else "  tidak ada")
print("Nilai y maksimum seluruh data:", full.loc[full.y.idxmax()].to_dict())

hourly = pd.DataFrame({"h": df.hour.iloc[te].values, "r": resid}).groupby("h")["r"].agg(["mean", "sem", "count"])
hourly["ci95_lo"] = hourly["mean"] - 1.96 * hourly["sem"]; hourly["ci95_hi"] = hourly["mean"] + 1.96 * hourly["sem"]
hourly["signif"] = (hourly.ci95_lo > 0) | (hourly.ci95_hi < 0)
print("\nResidual rata-rata per jam (RF, one-hot) dengan CI 95%:"); print(hourly.round(3))

cv.to_csv("cv_folds.csv", index=False); t_te.to_csv("test_metrics.csv"); hourly.to_csv("hourly_residual.csv")
pd.DataFrame({"ts": ts_te, "y": yte, **{k: v for k, v in p_te.items()}}).to_csv("test_predictions.csv", index=False)
print("\nTersimpan: test_metrics.csv, cv_folds.csv, hourly_residual.csv, test_predictions.csv")

# ---------------------------------------------------------------- Tambahan untuk naskah
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

section("H. PEMERIKSAAN: konfigurasi naskah lama (integer Hour/DoW, 80/20, 1.751 test)")
X_int = df[["hour", "dow", "lag1", "lag24"]]
o_tr, o_te = slice(0, n - 1751), slice(n - 1751, n)
for name, m in [("LR", LinearRegression()), ("RF", rf())]:
    m.fit(X_int.iloc[o_tr], df.y.iloc[o_tr]); r = metrics(df.y.iloc[o_te], m.predict(X_int.iloc[o_te]))
    print(name, {k: round(v, 3) for k, v in r.items()})
    if name == "LR": print("   koef [b0, hour, dow, lag1, lag24]:", np.round(np.r_[m.intercept_, m.coef_], 3))

section("I. KOEFISIEN LR (one-hot), FEATURE IMPORTANCE RF, SLOPE, DEVIASI LR")
lr = LinearRegression().fit(feats(["lag1", "lag24"]).iloc[trva], df.y.iloc[trva])
coef = pd.Series(lr.coef_, index=feats(["lag1", "lag24"]).columns)
hc, dc = coef.filter(like="hour_"), coef.filter(like="dow_")
print("intercept", round(lr.intercept_, 3), "| lag1", round(coef.lag1, 3), "| lag24", round(coef.lag24, 3))
print("hour dummies: min", round(hc.min(), 3), hc.idxmin(), "max", round(hc.max(), 3), hc.idxmax())
print("dow dummies:", dc.round(3).to_dict())
Xrf = feats(["lag1", "lag24"]); rfm = rf().fit(Xrf.iloc[trva], df.y.iloc[trva])

def grp(c):
    if c.startswith("hour_"): return "Hour"
    if c.startswith("dow_"): return "Day of Week"
    return {"lag1": "Lag-1", "lag24": "Lag-24"}[c]

mdi = pd.Series(rfm.feature_importances_, index=Xrf.columns).groupby(grp).sum() * 100
pim = permutation_importance(rfm, Xrf.iloc[te], df.y.iloc[te], n_repeats=10, random_state=42, n_jobs=-1)
perm = pd.Series(pim.importances_mean, index=Xrf.columns).groupby(grp).sum()
print("MDI (%):", mdi.round(1).to_dict()); print("Permutation (dMSE, test):", perm.round(3).to_dict())
slope = {k: np.polyfit(yte, p_te[k], 1)[0] for k in ("LR", "RF")}
print("Scatter slope:", {k: round(v, 3) for k, v in slope.items()})
dev = {}
for k in ("LR", "RF"):
    r_ = yte - p_te[k]; thr_k = 2 * t_va.loc[k, "RMSE"]; big = r_ > thr_k
    dev[k] = dict(MD=r_.mean(), MaxDev=r_.max(), MaxDev_ts=str(ts_te[np.argmax(r_)]), MinDev=r_.min(),
                  PosFrac=(r_ > 0).mean() * 100, Eplus=np.clip(r_, 0, None).sum(),
                  Eplus_pct=np.clip(r_, 0, None).sum() / yte.sum() * 100,
                  Thr=thr_k, N_thr=int(big.sum()), E_thr=r_[big].sum(), E_thr_pct=r_[big].sum() / yte.sum() * 100)
print(pd.DataFrame(dev))
posfreq = pd.DataFrame({"h": df.hour.iloc[te].values, "pos": resid > 0}).groupby("h").pos.mean() * 100
print("Frekuensi residual positif per jam (%):", posfreq.round(1).to_dict())
desc = dict(mean_all=df.y.mean(), test_mean=yte.mean(), test_std=yte.std(ddof=0),
            train_mean=df.y.iloc[tr].mean(), val_mean=df.y.iloc[va].mean(), test_total_kwh=yte.sum())
print("Statistik:", {k: round(v, 3) for k, v in desc.items()})

# ---------------------------------------------------------------- Gambar
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "grid.linestyle": ":", "savefig.dpi": 200})
C = {"Mean": "#9e9e9e", "Persistence": "#6d4c41", "LR": "#e53935", "RF": "#2e7d32",
     "RF_noLag1": "#81c784", "RF_calendar": "#1e5b8c"}
LBL = {"Mean": "Mean", "Persistence": "Persist.", "LR": "LR", "RF": "RF", "RF_noLag1": "RF\n−Lag1", "RF_calendar": "RF\ncal."}
TTL = {"LR": "Linear Regression", "RF": "Random Forest"}
order = list(t_te.index)

fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, (mt, ttl) in zip(ax, [("MAE", "MAE (kW) ↓"), ("RMSE", "RMSE (kW) ↓"), ("R2", "R² ↑"), ("CVRMSE", "CV(RMSE) (%) ↓")]):
    v = t_te[mt].values; a.bar(range(len(v)), v, color=[C[k] for k in order])
    a.set_xticks(range(len(v)), [LBL[k] for k in order], fontsize=8); a.set_title(ttl)
    for i, x in enumerate(v):
        a.annotate(f"{x:.0f}" if mt == "CVRMSE" else f"{x:.3f}", (i, x), xytext=(0, 2 if x >= 0 else -2),
                   textcoords="offset points", ha="center", va="bottom" if x >= 0 else "top", fontsize=7, fontweight="bold")
    if mt == "CVRMSE": a.axhline(30, color="k", ls="--", lw=1); a.text(5.4, 33, "ASHRAE G14: 30%", ha="right", fontsize=7.5)
    if mt == "R2": a.axhline(0, color="k", lw=.8)
fig.suptitle("Untouched test set (20 Oct – 31 Dec 2025, n = 1,748)", fontweight="bold"); fig.tight_layout()
fig.savefig("figures/fig3_performance.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, k in zip(ax, ("LR", "RF")):
    a.scatter(yte, p_te[k], s=6, alpha=.35, color=C[k]); lim = [0, 16]
    a.plot(lim, lim, "k--", lw=1, label="Ideal (y = x)")
    c = np.polyfit(yte, p_te[k], 1); a.plot(lim, np.polyval(c, lim), color=C[k], lw=2, label=f"Trend (slope = {c[0]:.3f})")
    a.set(xlabel="Actual active power (kW)", ylabel="Predicted (kW)", title=TTL[k], xlim=lim, ylim=lim); a.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig4_scatter.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, k in zip(ax, ("LR", "RF")):
    r_ = yte - p_te[k]; a.hist(r_, bins=50, color=C[k], alpha=.85, edgecolor="white")
    a.axvline(0, color="k", lw=1); a.axvline(r_.mean(), color="k", ls="--", lw=1, label=f"Mean = {r_.mean():.3f} kW")
    a.set(xlabel="Residual Y − Ŷ (kW)", ylabel="Frequency", title=TTL[k]); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig5_residual_hist.png"); plt.close(fig)

w = slice(0, 300)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
for a, k in zip(ax, ("LR", "RF")):
    a.plot(ts_te[w], yte[w], color="#424242", lw=1, label="Actual")
    a.plot(ts_te[w], p_te[k][w], color=C[k], lw=1.4, label=f"{k} prediction")
    a.set_ylabel("kW"); a.legend(fontsize=8, loc="upper right", ncol=2)
ax[1].set_xlabel("Test period (first 300 hours)"); fig.tight_layout(); fig.savefig("figures/fig6_timeseries.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
o = ["Day of Week", "Hour", "Lag-24", "Lag-1"]
ax[0].barh(o, mdi[o], color="#2e7d32"); ax[0].set(xlabel="MDI importance (%)", title="Mean decrease in impurity (training)")
for i, v in enumerate(mdi[o]): ax[0].text(v + .5, i, f"{v:.1f}%", va="center", fontsize=8)
ax[1].barh(o, perm[o], color="#1e5b8c"); ax[1].axvline(0, color="k", lw=.8)
ax[1].set(xlabel="Permutation importance (ΔMSE, kW²)", title="Permutation importance (test)")
for i, v in enumerate(perm[o]): ax[1].text(v, i, f" {v:.3f}", va="center", fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig7_importance.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar(hourly.index, hourly["mean"], color=np.where(hourly.signif.values, "#c62828", "#1e5b8c"),
          yerr=1.96 * hourly["sem"], capsize=2, ecolor="#555")
from matplotlib.patches import Patch
ax[0].axhline(0, color="k", lw=.8); ax[0].legend(handles=[Patch(color="#c62828", label="95% CI excludes 0"), Patch(color="#1e5b8c", label="95% CI includes 0")], fontsize=8)
ax[0].set(xlabel="Hour of day", ylabel="Mean residual Y − Ŷ (kW)", title="Mean hourly residual ± 95% CI (RF, test)")
ax[1].bar(posfreq.index, posfreq, color="#1e5b8c"); ax[1].axhline(50, color="r", ls="--", lw=1, label="50% reference")
ax[1].set(xlabel="Hour of day", ylabel="Hours with Y > Ŷ (%)", title="Positive-residual frequency by hour (RF, test)"); ax[1].legend(fontsize=8)
for a in ax: a.set_xticks(range(0, 24, 2))
fig.tight_layout(); fig.savefig("figures/fig8_hourly_residual.png"); plt.close(fig)

piv = cv.pivot(index="fold", columns="model", values="RMSE")[order]
labels = [f"F{f}\n{s:%d %b}–{e:%d %b}" for f, s, e in cv[["fold", "start", "end"]].drop_duplicates().itertuples(index=False)]
fig, ax = plt.subplots(figsize=(10, 3.8))
for k in order:
    ax.plot(piv.index, piv[k], marker="o", color=C[k], lw=2 if k in ("LR", "RF") else 1.2, label={"RF_noLag1": "RF without Lag-1", "RF_calendar": "RF calendar-only"}.get(k, k))
ax.set_xticks(piv.index, labels, fontsize=8); ax.set(ylabel="RMSE (kW)", title="Rolling-origin evaluation (expanding window, 720-h folds)")
ax.legend(fontsize=8, ncol=3); fig.tight_layout(); fig.savefig("figures/fig9_rolling_origin.png"); plt.close(fig)

agg_cols = ["MAE", "RMSE", "R2", "CVRMSE", "NMBE"]
out = dict(n=n, split=[i_tr, i_va - i_tr, n - i_va],
           split_dates={k: [str(df.ts.iloc[s].iloc[0]), str(df.ts.iloc[s].iloc[-1])] for k, s in [("train", tr), ("val", va), ("test", te)]},
           folds=[[str(s), str(e)] for s, e in cv[["start", "end"]].drop_duplicates().itertuples(index=False)],
           val=t_va.to_dict("index"), test=t_te.to_dict("index"),
           cv_mean=cv.groupby("model")[agg_cols].mean().to_dict("index"), cv_std=cv.groupby("model")[agg_cols].std().to_dict("index"),
           cv_rmse=piv.to_dict(), dm={f"{a}_vs_{b}": dm_test(yte, p_te[a], p_te[b]) for a, b in
                                      [("LR", "RF"), ("Persistence", "RF"), ("Persistence", "LR"), ("RF_noLag1", "RF"), ("RF_calendar", "LR")]},
           lr=dict(intercept=lr.intercept_, lag1=coef.lag1, lag24=coef.lag24, hour_min=hc.min(), hour_min_at=hc.idxmin(),
                   hour_max=hc.max(), hour_max_at=hc.idxmax(), dow=dc.to_dict()),
           mdi=mdi.to_dict(), perm=perm.to_dict(), slope=slope, dev=dev, hourly=hourly.reset_index().to_dict("list"),
           posfreq=posfreq.to_dict(), desc=desc)
json.dump(out, open("results.json", "w"), indent=1, default=float); print("\nTersimpan: results.json, figures/*.png")
