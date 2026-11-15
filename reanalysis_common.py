"""Shared building blocks for reanalysis_r2.py and reanalysis_hybrid.py.

Data loading (timestamps rounded to the hour), feature encodings, the 60/20/20 chronological split,
monthly expanding-window folds, metrics (incl. ASHRAE G14), Diebold-Mariano (HAC + HLN), CI helpers,
grouped permutation importance, a small markdown report writer, and the figure style.
"""
import random, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
LAGSETS = {"all": ["lag1", "lag24"], "noLag1": ["lag24"], "calendar": []}
MET = ["MAE", "RMSE", "R2", "CVRMSE", "NMBE"]
BENCH = ["Mean", "Persistence"]
RF_GRID = [dict(min_samples_leaf=msl, max_features=mf) for msl in (1, 5, 20, 50) for mf in (1.0, 0.5)]
MONTHS = pd.period_range("2025-04", "2025-12", freq="M")          # rolling-origin folds
SUBPERIODS = [("20–31 Oct", None, "2025-11-01"), ("November", "2025-11-01", "2025-12-01"),
              ("1–21 Dec", "2025-12-01", "2025-12-22"), ("22–31 Dec (holiday period)", "2025-12-22", None)]


def setup(seed=SEED):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    sys.stdout.reconfigure(encoding="utf-8")
    np.random.seed(seed); random.seed(seed)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)


# ------------------------------------------------------------------------------------------ report
class Report:
    """Accumulates markdown lines (also echoed to stdout)."""
    def __init__(self):
        self.lines = []

    def md(self, s=""):
        self.lines.append(s); print(s)

    @staticmethod
    def f3(x):
        if isinstance(x, (bool, np.bool_)): return "yes" if x else "no"
        if isinstance(x, (int, np.integer)): return f"{x:,}"
        if isinstance(x, (float, np.floating)): return "nan" if np.isnan(x) else f"{x:,.3f}"
        return str(x).replace("|", "\\|")

    def table(self, d, index=True):
        d = d.reset_index() if index else d
        hdr = "| " + " | ".join(map(str, d.columns)) + " |"
        isnum = lambda c: all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) for v in d[c])
        sep = "|" + "|".join("---:" if isnum(c) else "---" for c in d.columns) + "|"
        rows = ["| " + " | ".join(self.f3(v) for v in r) + " |" for r in d.itertuples(index=False)]
        self.md("\n".join([hdr, sep, *rows])); self.md()

    def save(self, path):
        open(path, "w", encoding="utf-8").write("\n".join(self.lines) + "\n")


# ------------------------------------------------------------------------------------------ data
def load_data(xlsx):
    """Columns A:B of the workbook -> hourly frame with rounded timestamps, hour, dow, lag1, lag24 (8,736 rows)."""
    raw = pd.read_excel(xlsx, sheet_name="Hasil Perhitungan Manual", usecols="A:B")
    raw.columns = ["ts", "y"]
    raw["ts"] = pd.to_datetime(raw["ts"]); raw["y"] = pd.to_numeric(raw["y"])
    n_raw = len(raw)
    ts_round = raw.ts.dt.round("h")
    n_jitter = int((ts_round != raw.ts).sum())
    n_hour_shift = int((ts_round.dt.hour != raw.ts.dt.hour).sum())   # rows where Excel HOUR() is one hour early
    raw["ts_excel"] = raw.ts; raw["ts"] = ts_round
    raw = raw.sort_values("ts").reset_index(drop=True)
    gaps = raw.ts.diff().dropna()
    assert n_raw == 8760 and raw.ts.is_unique and (gaps == pd.Timedelta("1h")).all() and raw.y.notna().all()
    df = raw.copy()
    df["hour"] = df.ts.dt.hour
    df["dow"] = df.ts.dt.day_of_week                                 # 0 = Monday
    df["lag1"] = df.y.shift(1)
    df["lag24"] = df.y.shift(24)
    df = df.dropna(subset=["lag1", "lag24"]).reset_index(drop=True)
    assert len(df) == 8736
    return dict(raw=raw, df=df, n_raw=n_raw, n_jitter=n_jitter, n_hour_shift=n_hour_shift)


def calendar(df, enc, column_order="numeric"):
    """Calendar features. one-hot reference levels: hour 0 and Monday.
    column_order='lexicographic' reproduces pd.get_dummies(df[['hour','dow']].astype(str), drop_first=True),
    i.e. the column order of the round-1 script (reanalysis.py)."""
    if enc == "int":
        return df[["hour", "dow"]].astype(float)
    if enc == "onehot":
        h = pd.get_dummies(pd.Categorical(df.hour, categories=range(24)), prefix="hour", drop_first=True)
        d = pd.get_dummies(pd.Categorical(df.dow, categories=range(7)), prefix="dow", drop_first=True)
        if column_order == "lexicographic":
            h = h[sorted(h.columns, key=lambda c: c.split("_")[1])]
        return pd.concat([h, d], axis=1).astype(float)
    if enc == "cyclic":
        return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * df.hour / 24), "hour_cos": np.cos(2 * np.pi * df.hour / 24),
                             "dow_sin": np.sin(2 * np.pi * df.dow / 7), "dow_cos": np.cos(2 * np.pi * df.dow / 7)})
    raise ValueError(enc)


def group_of(col):
    if col.startswith("hour"): return "Hour"
    if col.startswith("dow"): return "DayOfWeek"
    return {"lag1": "Lag-1", "lag24": "Lag-24"}[col]


def split_idx(n):
    """Chronological 60/20/20 -> (train, validation, test, train+validation) index arrays."""
    i_tr, i_va = int(0.6 * n), int(0.8 * n)
    a = np.arange(n)
    return a[:i_tr], a[i_tr:i_va], a[i_va:], a[:i_va]


def monthly_folds(df):
    """Expanding-window folds: for each month Apr–Dec 2025, train on everything before it."""
    for k, mo in enumerate(MONTHS, 1):
        fit_idx = np.where(df.ts < mo.start_time)[0]
        ev_idx = np.where((df.ts >= mo.start_time) & (df.ts <= mo.end_time))[0]
        yield k, mo, fit_idx, ev_idx


def bench_pred(df, fit_idx, ev_idx):
    return {"Mean": np.full(len(ev_idx), df.y.values[fit_idx].mean()), "Persistence": df.lag1.values[ev_idx]}


# ------------------------------------------------------------------------------------------ statistics
def metrics(y, p):
    """MAE, RMSE, R², CV(RMSE) %, NMBE % (= Σ(Y−Ŷ)/(n·Ȳ)), and the ASHRAE G14 hourly pass flag."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    rmse = mean_squared_error(y, p) ** 0.5; ybar = y.mean()
    cv = rmse / ybar * 100; nmbe = (y - p).sum() / (len(y) * ybar) * 100
    return dict(MAE=mean_absolute_error(y, p), RMSE=rmse, R2=r2_score(y, p), CVRMSE=cv, NMBE=nmbe,
                G14=bool(cv <= 30 and abs(nmbe) <= 10))


def dm_test(y, pa, pb, nw_lags=0):
    """Diebold-Mariano, squared loss. d = e_a^2 - e_b^2 (d>0 => B more accurate).
    Long-run variance by Newey-West (Bartlett) with `nw_lags`; 0 = classic h=1 DM.
    HLN small-sample factor for h=1, p from t(T-1)."""
    y = np.asarray(y, float); d = (y - pa) ** 2 - (y - pb) ** 2; T = len(d); dc = d - d.mean()
    lrv = dc @ dc / T
    for k in range(1, nw_lags + 1):
        lrv += 2 * (1 - k / (nw_lags + 1)) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T) * np.sqrt((T - 1) / T)
    return dict(mean_d=d.mean(), DM=stat, p=2 * stats.t.sf(abs(stat), T - 1))


def ci95(x):
    """mean, SD, and t-based 95 % CI of the mean."""
    x = np.asarray(x, float); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.std(ddof=1), x.mean() - h, x.mean() + h


def grouped_permutation(model, X, y, groups, n_repeats=10, seed=SEED):
    """Increase in MSE when all columns of a feature group are permuted jointly (rows shuffled together)."""
    rng = np.random.default_rng(seed)
    Xv = X.to_numpy().copy(); base = mean_squared_error(y, model.predict(Xv)); out = {}
    for g in pd.unique(np.asarray(groups)):
        cols = np.where(np.asarray(groups) == g)[0]; inc = []
        for _ in range(n_repeats):
            Xp = Xv.copy(); Xp[:, cols] = Xv[rng.permutation(len(Xv))][:, cols]
            inc.append(mean_squared_error(y, model.predict(Xp)) - base)
        out[g] = (np.mean(inc), np.std(inc, ddof=1))
    return out


def hourly_residual(hours, resid):
    g = pd.DataFrame({"h": hours, "r": resid}).groupby("h").r.agg(["mean", "sem", "count"])
    g["lo"], g["hi"] = g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"]
    g["CI excludes 0"] = (g.lo > 0) | (g.hi < 0)
    return g


# ------------------------------------------------------------------------------------------ figures
COL = {"LR": "#2a78d6", "RF": "#eb6834", "noLag1": "#1baf7a", "calendar": "#eda100", "Mean": "#8a8985",
       "Persistence": "#3d3c39", "extra": "#6250d6"}


def apply_style(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "grid.linestyle": ":", "savefig.dpi": 200,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
