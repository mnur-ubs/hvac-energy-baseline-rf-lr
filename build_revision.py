"""Buat naskah revisi dengan Track Changes dari hasil reanalysis.py (results.json + figures/)."""
import copy, json, re, zipfile, struct
from lxml import etree
from docx_track import Tracker, q, R

SRC = "../JORESD/REVISION/2026-A-Data-Driven-Energy-Baseline.docx"
DST = "../JORESD/REVISION/2026-A-Data-Driven-Energy-Baseline_R1-TrackChanges.docx"
AUTHOR, DATE = "Revision R1", "2026-10-28T12:00:00Z"
J = json.load(open("results.json"))
T = Tracker(AUTHOR, DATE)

zin = zipfile.ZipFile(SRC)
doc = etree.fromstring(zin.read("word/document.xml"))
rels = etree.fromstring(zin.read("word/_rels/document.xml.rels"))
body = doc.find(q("body"))
K = list(body)                         # indeks sesuai dump (P<i> / T<i>)
TPL_TABLE = copy.deepcopy(K[60])       # template tabel sebelum ada tanda revisi


def cell(ti, r, c, p=0):
    return K[ti].findall(q("tr"))[r].findall(q("tc"))[c].findall(q("p"))[p]


def ed(el, new, expect=None):
    """Edit paragraf; expect = potongan teks lama yang wajib ada (pengaman indeks)."""
    if expect: assert expect in T.repr(el), (expect, T.repr(el)[:120])
    T.edit(el, new)


f3 = lambda x: f"{x:.3f}".replace("-", "−")
f2 = lambda x: f"{x:.2f}".replace("-", "−")
te, va, cvm, cvs = J["test"], J["val"], J["cv_mean"], J["cv_std"]
dev, lr, mdi = J["dev"], J["lr"], J["mdi"]
dm = J["dm"]

# ======================================================================= ABSTRACT
ed(cell(8, 1, 1), (
    "Building energy management in commercial facilities increasingly demands adaptive, data-driven approaches to accurately "
    "characterize energy baselines. Heating, Ventilation, and Air Conditioning (HVAC) systems account for nearly 50% of total "
    "building energy demand in commercial buildings across Asia, yet conventional static threshold-based management proves "
    "inadequate for the highly volatile and non-linear nature of operational power consumption. This study evaluates a "
    "data-driven load-prediction baseline comparing Linear Regression (LR) and Random Forest (RF) algorithms applied to 8,760 "
    "hourly active power records extracted from a Building Automation System (BAS) spanning the full calendar year 2025 "
    "(8,736 modelling rows after lag construction). Four BAS-internal predictors were used: hour of day and day of week "
    "(one-hot encoded) and two autoregressive lag variables (Lag-1 and Lag-24). The data were split chronologically into "
    "training (60%), validation (20%) and an untouched test set (20%); models were selected on the validation set only, "
    "benchmarked against constant-mean and persistence predictors, and further assessed with five-fold rolling-origin "
    "validation and the Diebold–Mariano test. Models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error "
    "(RMSE), coefficient of determination (R²), CV(RMSE) and normalized mean bias error (NMBE). RF was selected on the "
    f"validation set (RMSE = {va['RF']['RMSE']:.3f} vs. {va['LR']['RMSE']:.3f} kW), but on the untouched test set LR achieved "
    f"MAE = {te['LR']['MAE']:.3f} kW, RMSE = {te['LR']['RMSE']:.3f} kW, and R² = {te['LR']['R2']:.3f}, compared to RF values "
    f"of MAE = {te['RF']['MAE']:.3f} kW, RMSE = {te['RF']['RMSE']:.3f} kW, and R² = {te['RF']['R2']:.3f} (Diebold–Mariano "
    f"p = {dm['LR_vs_RF'][1]:.3f}). Relative to a constant-mean predictor (RMSE = {te['Mean']['RMSE']:.3f} kW), LR improves "
    f"RMSE by only {100*(1-te['LR']['RMSE']/te['Mean']['RMSE']):.1f}% and RF by {100*(1-te['RF']['RMSE']/te['Mean']['RMSE']):.1f}%, "
    f"and CV(RMSE) of {te['LR']['CVRMSE']:.0f}–{te['RF']['CVRMSE']:.0f}% remains well above the 30% hourly threshold of "
    "ASHRAE Guideline 14. Rolling-origin validation shows that RF is the most accurate model from August to October but "
    "degrades sharply in November–December, and removing Lag-1 does not change RF test accuracy "
    f"(p = {dm['RF_noLag1_vs_RF'][1]:.2f}). Residual analysis on the test set identified {dev['RF']['N_thr']} hours whose "
    f"positive residual exceeded twice the validation RMSE, totalling {dev['RF']['E_thr']:.0f} kWh "
    f"({dev['RF']['E_thr_pct']:.1f}% of test-period consumption); because the baseline error is comparable to the mean load, "
    "these are candidate episodes for investigation rather than confirmed waste. BAS-only temporal and autoregressive "
    "features are therefore insufficient for an M&V-grade baseline in this building, which motivates the inclusion of "
    "weather and occupancy information."), "Building energy management")

# ======================================================================= INTRODUCTION
ed(K[15], (
    "A review of the existing literature identifies three critical research gaps: (1) most prior studies incorporate external "
    "sensor data (outdoor temperature, humidity, occupancy) creating sensor infrastructure dependencies unavailable in standard "
    "BAS deployments ⟦0⟧, ⟦1⟧; (2) direct LR-vs-RF comparisons specifically for HVAC baseline construction using exclusively "
    "BAS-internal temporal and autoregressive features remain scarce ⟦2⟧, ⟦3⟧; and (3) operational extension of ML baselines "
    "to quantify excess energy consumption and isolate anomalous events in real-world HVAC data is insufficiently explored "
    "⟦4⟧, ⟦5⟧, and is rarely accompanied by benchmark comparisons and out-of-sample validation. This study contributes: "
    "(1) development and evaluation of LR and RF models using only internally derived BAS features across 8,760 hourly "
    "records, with model selection on a separate validation split, constant-mean and persistence benchmarks, a Lag-1 "
    "ablation, rolling-origin validation and a Diebold–Mariano significance test; (2) comprehensive data exploration "
    "visualizations revealing operational patterns and justifying the feature engineering strategy; and (3) a residual-based "
    "deviation analysis on an untouched test set that quantifies excess energy above an error-calibrated threshold and the "
    "hourly residual structure of the baseline. This study provides a direct LR-vs-RF comparative evaluation for HVAC active "
    "power baseline construction using exclusively BAS-internal autoregressive and temporal features without external sensor "
    "dependency applied to a complete annual dataset (n = 8,760 hourly records) with subsequent operational "
    "deviation analysis, and reports transparently the extent to which such a baseline meets established M&V accuracy "
    "criteria. The remainder of this paper is organized as follows: Section 2 describes the dataset, feature engineering "
    "pipeline, validation protocol and modeling methodology; Section 3 presents and discusses the experimental results across "
    "all evaluation dimensions in relation to the literature; and Section 4 draws conclusions and identifies directions for "
    "future research. "), "three critical research gaps")

# ======================================================================= METHODS
ed(K[18], (
    "The analytical pipeline is illustrated in Figure 1, following a five-stage structure: data acquisition, preprocessing, "
    "feature engineering, model training and evaluation, and operational deployment. In Stage 4, the candidate models are "
    "compared on a separate validation split, and the model with the lower validation RMSE is adopted as the baseline before "
    "the test set is used; the test set is evaluated only once, after this choice has been fixed. No hyperparameter tuning "
    "was performed, so the selection reflects out-of-the-box performance on this dataset ⟦0⟧ ⟦1⟧."), "model selection criterion")

ed(K[22], (
    "The operational dataset was extracted from a BAS installed in a multi-story commercial office building in Indonesia. "
    "Active power consumption (kW) of the HVAC panel system was logged by a DPM at one-hour intervals covering January 1 to "
    "December 31, 2025. Following data quality assurance including null-value removal and timestamp normalization (the BAS "
    "export stored timestamps with sub-second floating-point offsets, which were rounded to the nearest hour) a final dataset "
    "of 8,760 complete, gap-free hourly records was retained. Table 1 presents the complete dataset specifications."),
   "operational dataset")

# Tabel 1
ed(cell(24, 1, 1), "Digital Power Meter (DPM) logged by the building BAS (hourly export)", "datalogBAS")
ed(cell(24, 4, 1), "8,760 rows (full annual cycle); 8,736 modelling rows after lag construction", "8,760 rows")
sd = J["split_dates"]
ed(cell(24, 7, 0), "Training set (60%)", "Training set")
ed(cell(24, 7, 1), f"{J['split'][0]:,} observations (2 Jan – 8 Aug 2025, chronological)", "7,009")
ed(cell(24, 8, 0), "Testing set (20%, untouched)", "Testing set")
ed(cell(24, 8, 1), f"{J['split'][2]:,} observations (20 Oct – 31 Dec 2025, chronological)", "1,751")
tr7 = K[24].findall(q("tr"))[7]
T.insert_row_after(tr7, ["Validation set (20%)", f"{J['split'][1]:,} observations (8 Aug – 20 Oct 2025, chronological)"])

ed(K[26], (
    "Four features were extracted from the BAS timestamp and historical power values, following the temporal-autoregressive "
    "design recommended for BAS-only energy prediction ⟦0⟧, ⟦1⟧: Hour (X1, 0–23), Day of Week (X2, 0–6), Lag-1 "
    "(X3 = Yt−1, kW), and Lag-24 (X4 = Yt−24, kW). Hour and Day of Week capture occupancy-driven diurnal and weekly cycles; "
    "Lag-1 captures short-term thermal inertia; Lag-24 captures daily seasonal autocorrelation. Hour and Day of Week were "
    "one-hot encoded (23 and 6 binary indicators, with 00:00 and Monday as reference categories), so that each hour and day "
    "can take its own level instead of a monotone linear trend; the same encoded feature matrix was supplied to both LR and "
    "RF to keep the comparison fair. Following lag construction, the first 24 observations were discarded, yielding 8,736 "
    f"complete modeling rows, which were split chronologically into training (60%, {J['split'][0]:,} rows), validation "
    f"(20%, {J['split'][1]:,} rows) and test (20%, {J['split'][2]:,} rows) sets. Table 2 provides the full feature schema."),
   "Four features were extracted")
for r in (3, 4):
    ed(cell(28, r, 1), "Categorical (one-hot)", "Integer")
ed(cell(28, 3, 2), "23 indicators (ref. 00:00)", "0–23")
ed(cell(28, 4, 2), "6 indicators (ref. Monday)", "0–6")

ed(K[31], "Ŷ = β₀ + Σₕ γₕHₕ + Σₖ δₖDₖ + β₁X₃ + β₂X₄ + ε    \t\t\t(1)", "β₀ + β₁X₁")
ed(K[32], (
    "where Hₕ (h = 1…23) and Dₖ (k = 1…6) are the one-hot hour and day-of-week indicators. The model fitted on the training "
    f"and validation data yielded β₀ = {lr['intercept']:.3f}, a Lag-1 coefficient β₁ = {f3(lr['lag1'])} and a Lag-24 "
    f"coefficient β₂ = {f3(lr['lag24'])}, while the hour indicators ranged from {f3(lr['hour_min'])} kW (12:00) to "
    f"+{lr['hour_max']:.3f} kW (20:00) relative to 00:00 and the day-of-week indicators from "
    f"{f3(min(lr['dow'].values()))} kW (Wednesday) to +{max(lr['dow'].values()):.3f} kW (Friday) relative to Monday. With "
    "one-hot encoding the diurnal cycle is represented by the hour indicators rather than by a single slope; the integer "
    "encoding used in the original submission forced a monotone increase over the hours of the day and a monotone decline "
    "from Monday to Sunday. The small Lag-24 coefficient indicates that OLS assigns little weight to the 24-hour lag once the "
    "hourly profile is included, suggesting that any contribution of the 24-hour pattern comes through non-linear "
    "interactions that the linear model cannot represent, which is examined with the RF feature importance in Section 3.3. "
    "Table 3 summarizes the coefficient set."), "The fitted model yielded")
ed(K[33], "Table ⟦0⟧⟦1⟧⟦2⟧3⟦3⟧. Estimated coefficients of the Linear Regression model with one-hot calendar encoding "
          "(fitted on training and validation data).", "Estimated coefficients")
t3 = [("Intercept", "β₀", f3(lr["intercept"]), "Expected load at 00:00 on Monday (lags = 0)"),
      ("Hour indicators", "γₕ (h = 1…23)", f"{f3(lr['hour_min'])} (12:00) to +{lr['hour_max']:.3f} (20:00)", "kW relative to 00:00"),
      ("Day-of-week indicators", "δₖ (k = 1…6)", f"{f3(min(lr['dow'].values()))} (Wed) to +{max(lr['dow'].values()):.3f} (Fri)",
       "kW relative to Monday"),
      ("Lag-1 coefficient", "β₁", f3(lr["lag1"]), "kW per unit prior-hour power"),
      ("Lag-24 coefficient", "β₂", f3(lr["lag24"]), "kW per unit same-hour prior-day power")]
for r, vals in enumerate(t3, start=1):
    for c, v in enumerate(vals):
        ed(cell(34, r, c), v)

ed(K[36], "Random Forest aggregates B = 100 bootstrap-trained decision trees (scikit-learn default settings, "
          "random_state = 42) via averaging ⟦0⟧:", "B = 100")
ed(K[40], (
    "where Nₜ is the proportion of training samples at node t and ΔD(t) is the variance reduction at that split. One-hot "
    "indicators are summed back to their parent feature, and permutation importance on the test set (mean increase in MSE "
    "over 10 random shuffles of a feature) is reported as a complementary check, because MDI is computed on training data "
    "and tends to favour continuous features. Implementation used Python scikit-learn ⟦0⟧."), "where Nₜ")
ed(K[42], "Performance was quantified using five regression metrics ⟦0⟧:", "three regression metrics")

eq7 = T.new_para(K[44], "CV(RMSE) = RMSE / Ȳ × 100%   \t\t\t\t\t\t(7)")
eq8 = T.new_para(K[44], "NMBE = Σᵢ(Yᵢ − Ŷᵢ) / (n·Ȳ) × 100%   \t\t\t\t\t(8)")
mv = T.new_para(K[40], (
    "CV(RMSE) and NMBE are the calibration statistics of ASHRAE Guideline 14, whose commonly applied hourly acceptance "
    "thresholds are CV(RMSE) ≤ 30% and |NMBE| ≤ 10%; they are reported so that the baseline can be judged against "
    "measurement and verification (M&V) practice. In M&V terms, the training and validation windows (2 January – 20 October "
    "2025) play the role of the baseline period and the test window (20 October – 31 December 2025) that of a "
    "pseudo-reporting period in which no energy-conservation measure was implemented, so no non-routine adjustments were "
    "applied. The study is therefore framed as a machine-learning load-prediction baseline evaluated against M&V accuracy "
    "criteria, not as a complete M&V savings determination."))
h_val = T.new_para(K[41], "Validation Protocol, Benchmarks and Significance Testing")
p_val = T.new_para(K[40], (
    f"The 8,736 modelling rows were split chronologically into training (60%; 2 January – 8 August 2025), validation (20%; "
    "8 August – 20 October 2025) and test (20%; 20 October – 31 December 2025) sets. Each candidate was trained on the "
    "training set and compared on the validation set; the selected configuration was then refitted on the training and "
    "validation data and evaluated once on the untouched test set. Four reference models were evaluated alongside LR and RF: "
    "a constant-mean predictor (training mean), which defines the improvement over a flat line; a persistence predictor "
    "Ŷ(t) = Y(t−1); an RF without Lag-1 (Hour, Day of Week, Lag-24); and a calendar-only RF (Hour, Day of Week). Robustness "
    "to the choice of evaluation window was assessed with rolling-origin (expanding-window) validation over five consecutive "
    "720-hour folds covering August–December 2025, with every model retrained on all data preceding each fold; the mean and "
    "standard deviation of each metric across folds are reported. Differences in test-set squared error between models were "
    "tested with the Diebold–Mariano (DM) test for one-step-ahead forecasts with the Harvey–Leybourne–Newbold small-sample "
    "correction; the word “significant” is used only where p < 0.05."))
T.insert_after(K[45], eq7, eq8, mv, h_val, p_val)

ed(K[47], "The predictions of the selected model on the untouched test set were used as a dynamic baseline to compute the "
          "following residual-based metrics ⟦0⟧, none of which was used for model selection:", "three operational metrics")
ed(K[48], "MD = (1/n) Σᵢ(Yᵢ − Ŷᵢ)   \t\t\t\t\t\t(9)", "MD =")
ed(K[49], "Max_Dev = max{Yᵢ − Ŷᵢ : Yᵢ > Ŷᵢ}   \t\t\t\t\t(10)", "Max_Dev")
ed(K[50], "PRF = (Σᵢ f(i) / n) × 100%, where f(i) = 1 if Yᵢ > Ŷᵢ   \t\t(11)", "%Overuse")
eq12 = T.new_para(K[49], "E⁺ = Σᵢ max(Yᵢ − Ŷᵢ, 0) · Δt   \t\t\t\t\t\t(12)")
eq13 = T.new_para(K[49], "E⁺τ = Σᵢ (Yᵢ − Ŷᵢ) · 𝟙[Yᵢ − Ŷᵢ > τ] · Δt,  τ = 2·RMSEval   \t\t(13)")
p_dev = T.new_para(K[40], (
    "MD (Eq. 9) measures model bias rather than wasted energy. The positive-residual fraction PRF (Eq. 11; the corrected form "
    "of the former %Overuse, whose condition was printed with the sign reversed) describes only the sign of the residuals "
    "and is close to 50% for any approximately unbiased model, so it is not interpreted as a waste rate. Energy-based "
    "quantities are given by Eq. (12), the cumulative positive residual energy with Δt = 1 h, and Eq. (13), which counts "
    "only residuals larger than τ, twice the validation RMSE of the model, as candidate excess-consumption episodes. The "
    "mean residual per hour of day is reported with a 95% confidence interval (±1.96 standard errors) to separate systematic "
    "hourly patterns from noise."))
T.insert_after(K[50], eq12, eq13, p_dev)

# ======================================================================= RESULTS — performance
ed(K[58], (
    "Table 4 presents the validation RMSE used for model selection and the metrics of all six models on the untouched "
    f"{J['split'][2]:,}-hour test set. On the validation set, RF clearly outperformed LR (RMSE = {va['RF']['RMSE']:.3f} vs. "
    f"{va['LR']['RMSE']:.3f} kW; R² = {va['RF']['R2']:.3f} vs. {f3(va['LR']['R2'])}) and was therefore selected as the "
    f"baseline. On the test set, however, the ranking reversed: LR achieved MAE = {te['LR']['MAE']:.3f} kW, "
    f"RMSE = {te['LR']['RMSE']:.3f} kW and R² = {te['LR']['R2']:.3f}, while RF achieved MAE = {te['RF']['MAE']:.3f} kW, "
    f"RMSE = {te['RF']['RMSE']:.3f} kW and R² = {te['RF']['R2']:.3f}. The DM test indicates that the lower squared error of "
    f"LR is statistically significant (DM = {f2(dm['LR_vs_RF'][0])}, p = {dm['LR_vs_RF'][1]:.3f}), so RF cannot be said "
    "to outperform LR on this test window. The calendar-only RF obtained the lowest test RMSE "
    f"({te['RF_calendar']['RMSE']:.3f} kW; R² = {te['RF_calendar']['R2']:.3f}), and the persistence benchmark was clearly the "
    f"worst model (RMSE = {te['Persistence']['RMSE']:.3f} kW), significantly worse than both LR and RF (p < 0.001). "),
   "Table 4 presents")
ed(K[59], "Table ⟦0⟧⟦1⟧⟦2⟧4⟦3⟧. Model performance: validation RMSE (used for selection), untouched test set "
          f"(n = {J['split'][2]:,}) and rolling-origin RMSE (mean ± SD over 5 folds).", "Global Model Performance")
rows4 = [["Model", "Val. RMSE (kW)", "Test MAE (kW)", "Test RMSE (kW)", "Test R²", "Test CV(RMSE) (%)", "Test NMBE (%)",
          "Rolling-origin RMSE (kW)"]]
names = {"Mean": "Constant mean", "Persistence": "Persistence", "LR": "Linear Regression", "RF": "Random Forest (selected)",
         "RF_noLag1": "RF without Lag-1", "RF_calendar": "RF calendar-only"}
for k, lab in names.items():
    t_ = te[k]
    rows4.append([lab, f"{va[k]['RMSE']:.3f}", f"{t_['MAE']:.3f}", f"{t_['RMSE']:.3f}", f3(t_["R2"]), f"{t_['CVRMSE']:.1f}",
                  f2(t_["NMBE"]),
                  f"{cvm[k]['RMSE']:.3f} ± {cvs[k]['RMSE']:.3f}"])
grid_w = sum(int(g.get(q("w"))) for g in K[60].find(q("tblGrid")))
w4 = [int(grid_w * x) for x in (.22, .10, .10, .10, .09, .12, .11, .16)]
new4 = T.new_table(TPL_TABLE, rows4, w4)
T.delete_table(K[60]); K[60].addnext(new4)

ed(K[62], (
    f"Figure 3 compares all models on the test set. Absolute R² values are low ({te['LR']['R2']:.3f} for LR and "
    f"{te['RF']['R2']:.3f} for RF): relative to a constant-mean predictor (RMSE = {te['Mean']['RMSE']:.3f} kW), LR reduces "
    f"RMSE by only {100*(1-te['LR']['RMSE']/te['Mean']['RMSE']):.1f}% and RF by "
    f"{100*(1-te['RF']['RMSE']/te['Mean']['RMSE']):.1f}%. CV(RMSE) ranges from {te['RF_calendar']['CVRMSE']:.0f}% to "
    f"{te['Persistence']['CVRMSE']:.0f}% across all models, more than three times the 30% hourly threshold of ASHRAE "
    f"Guideline 14, although NMBE ({te['RF']['NMBE']:.1f}% for RF and {te['LR']['NMBE']:.1f}% for LR) remains within the "
    "±10% bias limit. Because the baseline error remains comparable to the mean load, deviations of a similar magnitude "
    "cannot yet be reliably distinguished from model error ⟦0⟧, ⟦1⟧, ⟦2⟧. With the integer encoding of Hour and Day of Week "
    "and the 80/20 split of the original submission, the re-implemented pipeline reproduces the previously reported RF "
    "result (RMSE = 2.851 kW, R² = 0.084) and gives an LR RMSE of 2.936 kW; on that split, one-hot encoding lowers the "
    "LR RMSE to 2.794 kW but does not help RF (2.974 kW). The reversal of the ranking is therefore a consequence of the "
    "calendar encoding that the reviewers questioned, not of the change in data partitioning alone. "), "Figure 3 provides")
ed(K[64], "Figure ⟦0⟧⟦1⟧⟦2⟧3⟦3⟧. Test-set performance of all models (MAE, RMSE, R² and CV(RMSE); dashed line: ASHRAE "
          "Guideline 14 hourly threshold).", "Model Performance Comparison Dashboard")
ed(K[65], (
    "Scatter plots (Figure 4) provide the most direct diagnosis of model behavior. The LR prediction trend line on the test "
    f"set has a slope of only {J['slope']['LR']:.3f} nearly horizontal confirming systematic underfitting: the model assigns "
    "near-constant predictions regardless of actual load level, a failure mode that limits its usefulness as a dynamic "
    f"baseline. The RF trend slope of {J['slope']['RF']:.3f} shows more responsive prediction behavior, particularly in the "
    "0–8 kW operational range that constitutes the bulk of the test set; this responsiveness does not, however, translate "
    "into a lower squared error, because RF also issues high predictions at hours when the actual load is low."),
   "Scatter plots")
ed(K[68], (
    "Error distribution histograms (Figure 5) show that LR residuals peak at about −2 kW with a long positive tail "
    f"(maximum positive deviation +{dev['LR']['MaxDev']:.2f} kW, minimum {f2(dev['LR']['MinDev'])} kW), confirming "
    "systematic underprediction at peak loads. RF produces a more symmetric, zero-centered distribution (median residual "
    f"≈ 0 kW) but with heavy tails in both directions (maximum +{dev['RF']['MaxDev']:.2f} kW, maximum negative deviation "
    f"{f2(dev['RF']['MinDev'])} kW): it improves bias control as reported in the ensemble regression literature ⟦0⟧, ⟦1⟧, "
    "but at the cost of large overpredictions, which explains why its RMSE is higher despite a similar MAE. Time-series "
    "overlays (Figure 6) confirm visually that the LR prediction is comparatively flat, while the RF curve oscillates with "
    "the actual load dynamics."), "Error distribution histograms")
ed(K[74], (
    "The time-series overlay plots in Figure 6 provide an important complement to the scatter and histogram diagnostics. "
    "Viewing the model predictions alongside the actual power profile over a consecutive 300-observation window of the test "
    "set makes the LR failure mode immediately intuitive: the model's prediction trace is essentially a flat daily profile, "
    "unable to \"see\" the short-term load variation that drives the actual consumption up and down. The RF prediction trace, "
    "by contrast, moves with the actual data in approximate phase alignment, but not perfectly: it frequently anticipates "
    "high load at hours when consumption stays low. The visual responsiveness of RF is therefore not by itself evidence of a "
    "better baseline; on this test window the RF output would mask part of any genuine excess consumption, while the LR "
    "output would require the operator to account for the model's own systematic biases before interpreting any apparent "
    "deviations."), "The time-series overlay plots")

# ----------------------------------------------------------------------- feature importance
ed(K[76], (
    f"MDI feature importance scores (Figure 7, left; one-hot indicators summed per feature) give the hierarchy Lag-1 "
    f"({mdi['Lag-1']:.1f}%), Lag-24 ({mdi['Lag-24']:.1f}%), Hour ({mdi['Hour']:.1f}%), Day of Week ({mdi['Day of Week']:.1f}%), "
    "and permutation importance on the test set (Figure 7, right) also ranks Lag-1 first and Day of Week last. The two "
    f"autoregressive features together account for {mdi['Lag-1'] + mdi['Lag-24']:.1f}% of the RF's MDI weight. Thermal "
    "inertia the slow-responding thermal equilibrium of the chiller plant and building envelope makes the previous hour "
    "informative in principle ⟦0⟧, ⟦1⟧, ⟦2⟧. Importance, however, measures how much the fitted model uses a feature, not "
    "how much the feature improves out-of-sample accuracy: removing Lag-1 leaves the RF test RMSE unchanged "
    f"({te['RF_noLag1']['RMSE']:.3f} vs. {te['RF']['RMSE']:.3f} kW; DM p = {dm['RF_noLag1_vs_RF'][1]:.2f}), and the "
    f"calendar-only RF is more accurate than either ({te['RF_calendar']['RMSE']:.3f} kW)."), "MDI feature importance scores")
ed(K[79], "Figure ⟦0⟧⟦1⟧⟦2⟧7⟦3⟧. Feature importance of the Random Forest model: MDI on training data (left) and "
          "permutation importance on the test set (right).", "MDI feature importance scores for")

# ----------------------------------------------------------------------- deviation
d_, dl = dev["RF"], dev["LR"]
ed(K[81], (
    "Table 5 presents the deviation metrics for both models, computed on the untouched test set that was not used for model "
    f"selection. The mean deviation is small for both models ({dl['MD']:.3f} kW for LR and {d_['MD']:.3f} kW for RF), which "
    f"indicates low bias rather than wasted energy. The positive-residual fraction is {dl['PosFrac']:.1f}% for LR and "
    f"{d_['PosFrac']:.1f}% for RF; as expected for nearly unbiased models it is close to 50% and carries no information about "
    f"waste. In energy terms, the cumulative positive residual of RF amounts to {d_['Eplus']:,.0f} kWh "
    f"({d_['Eplus_pct']:.1f}% of the {J['desc']['test_total_kwh']:,.0f} kWh consumed in the test period), but this figure is "
    "an upper bound dominated by ordinary model error. Restricting the count to residuals larger than twice the validation "
    f"RMSE (τ = {d_['Thr']:.2f} kW for RF) leaves {d_['N_thr']} hours and {d_['E_thr']:,.0f} kWh ({d_['E_thr_pct']:.1f}% of "
    f"consumption) as candidate excess-consumption episodes, with a similar figure for LR ({dl['N_thr']} h, "
    f"{dl['E_thr']:,.0f} kWh). "), "Table 5 presents")
ed(K[82], "Table ⟦0⟧⟦1⟧⟦2⟧5⟦3⟧. Residual-based deviation metrics on the untouched test set "
          f"(n = {J['split'][2]:,}; total consumption {J['desc']['test_total_kwh']:,.1f} kWh)", "Operational deviation analysis")
fmt_ts = lambda s: re.sub(r"2025-(\d\d)-(\d\d) (\d\d):00:00", lambda m: f"{int(m[2])} {['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(m[1])]}, {m[3]}:00", s)
rows5 = [["Metric", "Linear Regression", "Random Forest", "Interpretation"],
         ["Mean Deviation, MD (kW)", f"{dl['MD']:.3f}", f"{d_['MD']:.3f}", "Model bias, not a waste quantity"],
         ["NMBE (%)", f"{te['LR']['NMBE']:.2f}", f"{te['RF']['NMBE']:.2f}", "Within the ±10% ASHRAE G14 bias limit"],
         ["Max positive residual (kW)", f"{dl['MaxDev']:.3f} ({fmt_ts(dl['MaxDev_ts'])})", f"{d_['MaxDev']:.3f} ({fmt_ts(d_['MaxDev_ts'])})",
          "≈3.3–4.4 × test RMSE; not corroborated, no fault attributed"],
         ["Positive-residual fraction, PRF (%)", f"{dl['PosFrac']:.2f}", f"{d_['PosFrac']:.2f}", "Residual sign only; ≈50% expected"],
         ["Cumulative positive residual, E⁺ (kWh)", f"{dl['Eplus']:,.1f} ({dl['Eplus_pct']:.1f}%)", f"{d_['Eplus']:,.1f} ({d_['Eplus_pct']:.1f}%)",
          "Upper bound; dominated by model error"],
         ["Threshold τ = 2·RMSEval (kW)", f"{dl['Thr']:.3f}", f"{d_['Thr']:.3f}", "Error-calibrated detection threshold"],
         ["Hours above τ", f"{dl['N_thr']}", f"{d_['N_thr']}", "Candidate episodes for investigation"],
         ["Energy above τ, E⁺τ (kWh)", f"{dl['E_thr']:,.1f} ({dl['E_thr_pct']:.1f}%)", f"{d_['E_thr']:,.1f} ({d_['E_thr_pct']:.1f}%)",
          "Candidate excess consumption (% of test-period total)"]]
grid_w5 = sum(int(g.get(q("w"))) for g in K[83].find(q("tblGrid")))
new5 = T.new_table(TPL_TABLE, rows5, [int(grid_w5 * x) for x in (.28, .20, .20, .32)])
T.delete_table(K[83]); K[83].addnext(new5)

hr = J["hourly"]; sig_pos = [h for h, m, s in zip(hr["h"], hr["mean"], hr["signif"]) if s and m > 0]
sig_neg = [h for h, m, s in zip(hr["h"], hr["mean"], hr["signif"]) if s and m < 0]
hh = lambda hs: ", ".join(f"{h:02d}:00" for h in hs)
ed(K[84], (
    f"The maximum positive residual of the RF baseline, {d_['MaxDev']:.3f} kW, occurred on 12 December 2025 at 00:00, when "
    "10.37 kW was recorded against a prediction of 0.62 kW, and most of the largest residuals fall in December at night or "
    "in the early evening. The 8.890 kW event reported in the original submission was likewise a model residual rather than "
    "a measured load. No maintenance log, alarm record or chiller trend was available to corroborate a mechanical fault at "
    f"these timestamps, so the earlier fault attribution has been withdrawn; at about {d_['MaxDev']/te['RF']['RMSE']:.1f} "
    "times the test RMSE, residuals of this size are compatible with model error on a volatile, bimodal load. The hourly "
    "residual profile (Figure 8), computed after one-hot encoding of Hour, still shows positive mean residuals whose 95% "
    f"confidence intervals exclude zero at {hh(sig_pos)} (about 1.0–1.4 kW) and negative ones at {hh(sig_neg)}. Because the "
    "pattern persists after re-encoding, it is not solely an artefact of the integer encoding; it indicates that night-time "
    "and evening consumption in the test period, concentrated in November–December, was higher than the baseline learned "
    "from earlier months."), "The maximum deviation of 8.890 kW")
ed(K[86], "Figure ⟦0⟧⟦1⟧⟦2⟧8⟦3⟧.  Left: mean residual per hour of day (RF baseline, test set) with 95% confidence "
          "intervals; red bars mark intervals that exclude zero. Right: percentage of hours with a positive residual.",
   "average hourly deviation profile")

# ----------------------------------------------------------------------- robustness (rolling-origin)
h_rob = T.new_para(K[80], "Robustness Across Evaluation Windows")
K[86].addnext(h_rob)
cr = J["cv_rmse"]
ed(K[87], (
    "Figure 9 reports the RMSE of all models in each of the five rolling-origin folds. Averaged over the folds, RF has the "
    f"lowest mean RMSE ({cvm['RF']['RMSE']:.3f} ± {cvs['RF']['RMSE']:.3f} kW), followed by the calendar-only RF "
    f"({cvm['RF_calendar']['RMSE']:.3f} ± {cvs['RF_calendar']['RMSE']:.3f} kW) and LR ({cvm['LR']['RMSE']:.3f} ± "
    f"{cvs['LR']['RMSE']:.3f} kW), but RF is also by far the least stable model. RF is clearly the most accurate model from "
    f"August to October (fold RMSE {min(cr['RF'][str(i)] for i in (1, 2, 3)):.2f}–{max(cr['RF'][str(i)] for i in (1, 2, 3)):.2f} kW) "
    f"and then deteriorates sharply in November ({cr['RF']['4']:.3f} kW, worse than the constant mean at "
    f"{cr['Mean']['4']:.3f} kW) and December ({cr['RF']['5']:.3f} kW), whereas LR and the calendar-only RF change little "
    "across folds. "), "Figure 9 consolidates")
ed(K[89], "Figure ⟦0⟧⟦1⟧⟦2⟧9⟦3⟧. Rolling-origin (expanding-window) RMSE of all models over five consecutive 720-hour "
          "folds (August–December 2025).", "Comprehensive performance")
ed(K[91], (
    "These results address whether the evaluation window, rather than the models, explains the poor test performance. The "
    "November–December period is the hardest to predict for every model, but it penalizes RF disproportionately: the test "
    "window coincides with a change in operating behavior (higher night-time and evening consumption, Figure 8) that the "
    "autoregressive RF extrapolates poorly from earlier months, while the calendar-based models are more stable. The "
    "validation-based choice of RF, made on August–October data, did not generalize to this regime, which is why the "
    "untouched test set and the rolling-origin evaluation are reported alongside the validation result."),
   "Reading the chart from left to right")

# ----------------------------------------------------------------------- discussion
ed(K[93], (
    "The ensemble-over-linear advantage documented in ⟦0⟧, ⟦1⟧, and ⟦2⟧ for volatile building loads is visible in this "
    "building only on the validation set and in the August–October folds; on the test window it disappears once the "
    "calendar features are one-hot encoded. The failure mode of LR is nonetheless structurally visible in the scatter plot: "
    f"a trend-line slope of just {J['slope']['LR']:.3f} effectively horizontal reveals that OLS converges to a near-constant "
    "prediction biased toward the high-density low-load region of the power distribution, systematically underpredicting "
    "peak events at the cost of practical baseline utility ⟦3⟧, ⟦4⟧. Its lower RMSE comes from avoiding the large "
    "overpredictions of RF, not from tracking the load better. The small Lag-24 coefficient "
    f"(β₂ = {f3(lr['lag24'])}) offers a further diagnostic. RF assigns this feature {mdi['Lag-24']:.1f}% MDI weight, yet "
    "OLS finds little marginal value in it once the hourly profile is included a divergence consistent with Lag-24's "
    "contribution being non-linearly conditional on other features, exactly the kind of interaction decision trees represent "
    "through sequential conditional splits but OLS cannot ⟦5⟧, ⟦6⟧."), "The 7.3% lower MAE")
ed(K[94], (
    f"It is also worth clarifying what the low R² values ({te['RF']['R2']:.3f}–{te['LR']['R2']:.3f} on the test set) mean in "
    "this context. Deviation detection requires the baseline's own error to be small relative to the deviations being "
    "sought. The BAS-only constraint deliberately excludes occupancy and environmental inputs that drive most short-term "
    f"stochastic variance ⟦0⟧, ⟦1⟧, and the remaining error (RMSE ≈ {te['LR']['RMSE']:.1f}–{te['RF']['RMSE']:.1f} kW against "
    f"a test-period mean load of {J['desc']['test_mean']:.2f} kW) is of the same size as the load itself, so a genuine excess "
    "of 2 kW cannot be distinguished from model error. A low R² is therefore not harmless for this application: neither LR "
    "nor RF, in their current form, provides a baseline suitable for attributing individual deviations to operational "
    "causes."), "It is also worth clarifying")
ed(K[95], "Thermal Inertia and the Limits of Autoregressive Features", "Thermal Inertia")
ed(K[96], (
    f"The MDI hierarchy Lag-1 ({mdi['Lag-1']:.1f}%), Lag-24 ({mdi['Lag-24']:.1f}%), Hour ({mdi['Hour']:.1f}%), Day of Week "
    f"({mdi['Day of Week']:.1f}%) has a plausible physical interpretation. Large commercial chiller systems operate on "
    "hour-scale thermal time constants: compressor loading state, chilled-water circuit temperatures, and refrigerant "
    "conditions from the prior hour carry forward-predictive information because these system properties change slowly "
    "⟦0⟧ ⟦1⟧, ⟦2⟧. In this dataset, however, that information does not improve out-of-sample accuracy: the RF without "
    "Lag-1 is as accurate as the full RF on the test set, the calendar-only RF is more accurate than both, and the "
    "persistence predictor, which uses Lag-1 alone, is the worst model in every fold. Lopez-Villamor et al. ⟦3⟧ and Liu et "
    "al. ⟦4⟧ reported strong autoregressive effects for commercial thermal loads and HVAC systems; the present result shows "
    "that such effects are building-specific and should be verified by ablation rather than inferred from importance "
    "scores. Lag-1 also works against the purpose of a baseline: because it is the previous hour's actual consumption, an "
    "anomaly that persists beyond one hour is absorbed into the prediction, which is why M&V baselines usually exclude the "
    "dependent variable's own recent history. For BAS-only deployments, the implication is that richer calendar information "
    "(holiday and operating-schedule indicators) and, where possible, outdoor temperature are more promising than "
    "additional lags ⟦5⟧, ⟦6⟧."), "The MDI hierarchy")
ed(K[98], (
    "Given the current error level, the hourly residual pattern should not be used to target BAS schedule recalibration or "
    "maintenance at specific hours, and the hour-specific recommendations of the original submission have been withdrawn. "
    "What the analysis does support is narrower: during November–December the building consumed more at night and in the "
    "early evening than its January–October behavior would predict, and the "
    f"{d_['N_thr']} hours above the 2×RMSE threshold provide a short, prioritized list of timestamps that a facility manager "
    "can check against BAS schedules, alarm logs and chiller trends ⟦0⟧, ⟦1⟧. Whether these episodes reflect overcooling, a "
    "schedule change, or legitimate year-end operation cannot be determined from the active power data alone ⟦2⟧, ⟦3⟧."),
   "The 51.77% overuse rate")
ed(K[99], (
    "The same caution applies to individual extreme residuals. Static threshold monitoring cannot distinguish them from the "
    "natural volatility of the HVAC profile, and the present baseline cannot either: the largest RF residuals (9.1–9.7 kW) "
    "are about three times the test RMSE and, without independent corroboration, should be treated as candidates for "
    "inspection rather than as faults ⟦0⟧. The higher maximum residual of the LR baseline "
    f"({dl['MaxDev']:.2f} kW) likewise reflects its structural underprediction at peak-load hours."), "The maximum deviation of 8.890 kW")
ed(K[101], (
    "Four limitations apply. First, excluding ambient temperature and occupancy data the deliberate BAS-only constraint "
    "leaves the majority of short-term stochastic variance unexplained; incorporating even basic outdoor temperature data "
    "would be expected to improve R2 meaningfully ⟦0⟧, ⟦1⟧, ⟦2⟧. Second, default Scikit-learn hyperparameters were used "
    "throughout without cross-validation tuning, which may leave performance improvement on the table ⟦3⟧. Third, the "
    "quantitative results feature importance rankings, residual statistics, and deviation magnitudes are specific to this "
    "building, climate, and annual measurement window; direct numerical transfer to other building types or non-tropical "
    "climates should not be assumed, though the methodological findings (the need for benchmark models, ablation and "
    "out-of-sample validation before interpreting deviations) are expected to apply generally ⟦4⟧, ⟦5⟧. Fourth, a single "
    "year of data means that the test window necessarily coincides with year-end operation; the rolling-origin analysis "
    "shows how performance varies across months, but a multi-year dataset would be required to separate seasonal regime "
    "changes from model weaknesses. Future work should incorporate outdoor temperature and holiday calendars, explore LSTM "
    "and GRU architectures for capturing longer autoregressive dependencies ⟦6⟧, ⟦7⟧, probabilistic prediction intervals to "
    "quantify baseline uncertainty in operational deployment ⟦8⟧, and multi-building validation studies to establish the "
    "scalability of the proposed BAS-native pipeline ⟦9⟧."), "Three limitations apply")

# ======================================================================= CONCLUSIONS
ed(K[103], (
    "This study evaluated a BAS-native, sensor-free HVAC load-prediction baseline using exclusively active power time-series "
    "data, with model selection on a validation split, an untouched test set, constant-mean and persistence benchmarks, a "
    "Lag-1 ablation, rolling-origin validation and a Diebold–Mariano test. Three conclusions follow, each conditional on this "
    "validation design and on a single building-year. First, RF was preferred on the validation set "
    f"(RMSE = {va['RF']['RMSE']:.3f} vs. {va['LR']['RMSE']:.3f} kW) but did not outperform LR on the untouched test set, "
    f"where LR was more accurate (RMSE = {te['LR']['RMSE']:.3f} vs. {te['RF']['RMSE']:.3f} kW; DM p = {dm['LR_vs_RF'][1]:.3f}); "
    "across five rolling-origin folds RF was the best model from August to October and the least stable overall "
    f"(RMSE = {cvm['RF']['RMSE']:.3f} ± {cvs['RF']['RMSE']:.3f} kW). The scatter plot slope evidence "
    f"({J['slope']['RF']:.3f} for RF vs. {J['slope']['LR']:.3f} for LR) confirms that LR systematically underfits the "
    "volatile HVAC load profile, but responsiveness alone did not make RF the more accurate baseline. Second, the "
    "four-feature temporal-autoregressive set (Lag-1, Lag-24, Hour, Day of Week) is not sufficient for an M&V-grade baseline "
    f"in this building: the best model improves RMSE by only {100*(1-te['RF_calendar']['RMSE']/te['Mean']['RMSE']):.1f}% over a "
    "constant mean, CV(RMSE) remains near 100% against the 30% threshold of ASHRAE Guideline 14, and Lag-1, although "
    f"dominant in MDI importance ({mdi['Lag-1']:.1f}%), adds no out-of-sample accuracy. Third, a residual analysis on the "
    f"untouched test set identifies {d_['N_thr']} hours ({d_['E_thr']:,.0f} kWh, {d_['E_thr_pct']:.1f}% of test-period "
    "consumption) with residuals above twice the validation RMSE, concentrated at night and in the early evening in "
    "November–December; these are candidates for investigation, not confirmed waste or faults, and hour-specific "
    "operational recommendations are withheld until the baseline meets a usable error threshold. Incorporating outdoor "
    "temperature, holiday and operating-schedule information, and evaluating over multiple years, are the necessary next "
    "steps before such a baseline can support operational decisions."), "This study demonstrates the feasibility")

# Data availability statement sebelum REFERENCES
h_das = T.new_para(K[104], "DATA AVAILABILITY STATEMENT")
p_das = T.new_para(K[103], (
    "The hourly HVAC active power data were obtained from the Building Automation System of a commercial office building in "
    "Indonesia and are not publicly available because of the building operator's confidentiality requirements. The data and "
    "the Python analysis scripts (data preparation, model training, rolling-origin validation and Diebold–Mariano test) are "
    "available from the corresponding author upon reasonable request and with the permission of the building operator."))
T.insert_after(K[103], h_das, p_das)

# Caption Figure 1 diketik manual (bukan field SEQ), sehingga Word menomori grafik time-series sebagai
# "Figure 1" dan semua gambar berikutnya bergeser satu. Ganti angka "1" dengan field SEQ Figure.
runs56 = [c for c in K[56] if c.tag == q("r")]
i0 = next(i for i, r in enumerate(runs56) if r.find(q("fldChar")) is not None and r.find(q("fldChar")).get(q("fldCharType")) == "begin")
i1 = next(i for i, r in enumerate(runs56) if r.find(q("fldChar")) is not None and r.find(q("fldChar")).get(q("fldCharType")) == "end")
seq = [copy.deepcopy(r) for r in runs56[i0:i1 + 1]]
for r in seq:
    if r.find(q("fldChar")) is None and r.find(q("instrText")) is None:
        r.find(q("t")).text = "1"
ed(K[20], "Figure . Research methodology flowchart covering five stages: data acquisition, preparation, modeling, "
          "evaluation, and deployment", "Figure 1. Research methodology")
d1 = next(d for d in K[20].iter(q("del")) if "".join(d.itertext()) == "1")
d1.addnext(T._wrap("ins", seq))

# ======================================================================= FIGURES
def png_size(path):
    with open(path, "rb") as f:
        f.read(16); return struct.unpack(">II", f.read(8))

media, nid = {}, 2000
figmap = {63: "fig3_performance", 66: "fig4_scatter", 70: "fig5_residual_hist", 72: "fig6_timeseries",
          78: "fig7_importance", 85: "fig8_hourly_residual", 88: "fig9_rolling_origin"}
for pi, name in figmap.items():
    rid = f"rIdR1{name.split('_')[0]}"; target = f"media/r1_{name}.png"
    rel = etree.SubElement(rels, "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    rel.set("Id", rid); rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"); rel.set("Target", target)
    media["word/" + target] = open(f"figures/{name}.png", "rb").read()
    T.replace_image(K[pi], rid, png_size(f"figures/{name}.png"), nid, min_width_emu=int(15.5 / 2.54 * 914400)); nid += 1

# ======================================================================= WRITE
zout = zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == "word/document.xml":
        data = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
    elif it.filename == "word/_rels/document.xml.rels":
        data = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)
    zout.writestr(it, data)
for k, v in media.items(): zout.writestr(k, v)
zout.close()
print("Tersimpan:", DST, "| revisi id terakhir:", T.next_id)
