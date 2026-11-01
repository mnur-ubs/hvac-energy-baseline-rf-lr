"""Isi placeholder di Draft-Response-to-Reviewer.docx dengan hasil analisis ulang (tanpa Track Changes)."""
import copy, json, re, zipfile
from lxml import etree
from docx_track import q

SRC = "../JORESD/REVISION/Draft-Response-to-Reviewer.docx"
DST = "../JORESD/REVISION/Draft-Response-to-Reviewer_R1.docx"
J = json.load(open("results.json"))
te, va, cvm, cvs, dev, dm, mdi = J["test"], J["val"], J["cv_mean"], J["cv_std"], J["dev"], J["dm"], J["mdi"]
imp = lambda k: 100 * (1 - te[k]["RMSE"] / te["Mean"]["RMSE"])

RESP = {
 1: (f"We thank the reviewer for this important observation and agree that the improvement over a constant-mean predictor "
     f"and the CV(RMSE) must be reported transparently. After the re-analysis (one-hot calendar encoding, 60/20/20 "
     f"chronological split with selection on validation, untouched test set), the abstract now states that, relative to a "
     f"constant-mean predictor (test RMSE = {te['Mean']['RMSE']:.3f} kW), LR improves RMSE by only {imp('LR'):.1f}% and RF by "
     f"{imp('RF'):.1f}%, and that CV(RMSE) is {te['LR']['CVRMSE']:.0f}–{te['RF']['CVRMSE']:.0f}%, far above the ~30% hourly "
     "threshold of ASHRAE Guideline 14. CV(RMSE) and NMBE are now defined in Eqs. (7)–(8) and reported for all models in "
     "Table 4. For transparency, we note that re-running the original pipeline reproduces the RF figures quoted by the "
     "reviewer (RMSE = 2.851 kW, R² = 0.084).",
     "Abstract; Section 2.5 (Eqs. 7–8 and new paragraph on ASHRAE Guideline 14); Section 3.2; Table 4; Figure 3."),
 2: ("We agree. The word “significantly” has been removed. A Diebold–Mariano test (squared-error loss, h = 1, "
     "Harvey–Leybourne–Newbold correction) was added on the untouched test set. The difference between the two models is "
     f"significant, but in favour of LR (DM = −{abs(dm['LR_vs_RF'][0]):.2f}, p = {dm['LR_vs_RF'][1]:.3f}), so the revised "
     "manuscript no longer claims that RF outperforms LR. Persistence is significantly worse than both models (p < 0.001), "
     f"and removing Lag-1 does not change RF accuracy (p = {dm['RF_noLag1_vs_RF'][1]:.2f}). “Significant” is now "
     "used only where p < 0.05.",
     "Abstract; Section 2.6 (new); Section 3.2; Conclusions (first conclusion)."),
 3: ("We appreciate this clarification. We have adopted one consistent framing: the study is now presented as a "
     "machine-learning load-prediction baseline evaluated against M&V accuracy criteria, not as a complete M&V savings "
     "determination. Section 2.5 now states explicitly how the two vocabularies map onto each other: the training and "
     "validation windows (2 January – 20 October 2025) correspond to the baseline period, and the untouched test window "
     "(20 October – 31 December 2025) to a pseudo-reporting period in which no energy-conservation measure was implemented, "
     "so no non-routine adjustments were applied. The title has been retained, with “energy baseline” used in the "
     "sense of an expected-consumption reference.",
     "Abstract; Section 2.5 (new paragraph); Section 3.2 (comparison with ASHRAE Guideline 14)."),
 6: ("We thank the reviewer; this point proved decisive. Hour and Day of Week are now one-hot encoded (23 and 6 indicators, "
     "with 00:00 and Monday as references), and the same encoded matrix is used for LR and RF. The revised LR coefficients "
     f"are summarised in Table 3 (hour effects from −{abs(J['lr']['hour_min']):.3f} kW at 12:00 to +{J['lr']['hour_max']:.3f} kW "
     "at 20:00 relative to 00:00). The re-encoding changes the conclusion: on the same 80/20 split as the original "
     "submission, the LR RMSE falls from 2.936 to 2.794 kW while RF does not improve (2.851 → 2.974 kW), and on the new "
     f"untouched test set LR (RMSE = {te['LR']['RMSE']:.3f} kW) is more accurate than RF ({te['RF']['RMSE']:.3f} kW). The "
     "statements about LR's structural limitations have been revised accordingly; the scatter-slope diagnosis is retained "
     f"with the updated values ({J['slope']['LR']:.3f} for LR vs. {J['slope']['RF']:.3f} for RF).",
     "Section 2.3; Eq. (1) and Section 2.4; Tables 2, 3 and 4; Sections 3.2 and 3.6; Figures 3–6."),
 7: ("We agree with this analysis. On checking the calculation we found that the computation itself counted hours with "
     "Yᵢ > Ŷᵢ; only the condition printed in Eq. (9) had the sign reversed. The metric is now written correctly (Eq. 11) and "
     "renamed the positive-residual fraction (PRF). It is described explicitly as a residual-sign statistic (≈50% for an "
     f"unbiased model; {dev['RF']['PosFrac']:.1f}% for RF on the test set) and is no longer interpreted as waste. Mean "
     "Deviation (Eq. 9) is now described as model bias. Two energy-based measures have been added: the cumulative positive "
     "residual energy E⁺ = Σ max(Y − Ŷ, 0)·Δt (Eq. 12) and the energy of residuals above τ = 2 × validation RMSE (Eq. 13). "
     f"For RF these amount to {dev['RF']['Eplus']:,.0f} kWh ({dev['RF']['Eplus_pct']:.1f}% of test-period consumption, an "
     f"upper bound dominated by model error) and {dev['RF']['E_thr']:,.0f} kWh in {dev['RF']['N_thr']} hours "
     f"({dev['RF']['E_thr_pct']:.1f}%), respectively; the latter are reported as candidate episodes for investigation, not "
     "as confirmed waste.",
     "Section 2.7 (Eqs. 9–13 and new paragraph); Section 3.4; Table 5; Abstract; Conclusions (third conclusion)."),
 8: ("We thank the reviewer for this suggestion. A five-fold rolling-origin evaluation with 720-hour folds "
     "(August–December 2025) was added (Figure 9, Table 4). RF is the most accurate model from August to October (fold "
     f"RMSE {min(J['cv_rmse']['RF'][str(i)] for i in (1, 2, 3)):.2f}–{max(J['cv_rmse']['RF'][str(i)] for i in (1, 2, 3)):.2f} kW) "
     f"but deteriorates sharply in November ({J['cv_rmse']['RF']['4']:.3f} kW, worse than the constant mean at "
     f"{J['cv_rmse']['Mean']['4']:.3f} kW) and December ({J['cv_rmse']['RF']['5']:.3f} kW), whereas LR and the calendar-only "
     "RF remain stable (≈2.6–2.9 kW). The November–December period is therefore the hardest for every model and penalises "
     "RF disproportionately: the test window coincides with a change in operating behaviour (higher night-time and evening "
     "consumption), which explains a large part of the poor test performance. This is now discussed explicitly, and a "
     "fourth limitation (single year of data) has been added.",
     "Section 2.6; Section 3.5 (new) and Figure 9; Section 3.9 (fourth limitation)."),
 9: (f"We agree. The data are now split chronologically into training (60%, {J['split'][0]:,} h), validation (20%, "
     f"{J['split'][1]:,} h) and test (20%, {J['split'][2]:,} h) sets. The model choice (lower validation RMSE) is made on "
     f"the validation set only (RF: {va['RF']['RMSE']:.3f} kW vs. LR: {va['LR']['RMSE']:.3f} kW); the selected configuration "
     "is then refitted on the training and validation data and evaluated once on the untouched test set. The test result "
     "does not confirm the validation choice (Table 4), and we report this openly.",
     "Section 2.1; Section 2.6 (new); Table 1; Section 3.2; Table 4."),
 10: ("We agree. Rolling-origin (expanding-window) validation with five folds of 720 h was added, and the mean ± SD of MAE, "
      "RMSE, R², CV(RMSE) and NMBE were computed for all six models (RMSE: RF "
      f"{cvm['RF']['RMSE']:.3f} ± {cvs['RF']['RMSE']:.3f} kW, calendar-only RF {cvm['RF_calendar']['RMSE']:.3f} ± "
      f"{cvs['RF_calendar']['RMSE']:.3f} kW, LR {cvm['LR']['RMSE']:.3f} ± {cvs['LR']['RMSE']:.3f} kW). The RMSE spread is "
      "reported in Table 4 and the per-fold results in Figure 9. Together with the Diebold–Mariano test (comment 2), this "
      "shows that the original 2.4% RMSE difference was not evidence of superiority.",
      "Section 2.6; Table 4; Section 3.5 and Figure 9."),
 11: (f"We thank the reviewer for spotting this. After lag construction 8,736 rows remain; all splits are now derived from "
      f"these rows ({J['split'][0]:,} + {J['split'][1]:,} + {J['split'][2]:,} = 8,736), and Table 1 reports both the 8,760 "
      "raw records and the 8,736 modelling rows. During the re-analysis we also found that the BAS export stored timestamps "
      "with sub-second floating-point offsets (e.g., 00:59:59.999 instead of 01:00), which shifted the hour of day for part "
      "of the records; timestamps are now rounded to the nearest hour, and this step is described in Section 2.2.",
      "Table 1; Sections 2.2 and 2.3."),
 12: ("We agree. A Data Availability Statement has been added: the data are confidential to the building operator and are "
      "available from the corresponding author upon reasonable request with the operator's permission, together with the "
      "Python analysis scripts. The internal file name has been removed from Table 1. Because the Streamlit dashboard is a "
      "prototype that is not part of the evaluated method, it has been removed as a contribution of this paper.",
      "Data Availability Statement (new, before the References); Table 1; Introduction (list of contributions)."),
 13: ("We agree. The argument that low R² is unimportant has been replaced (Section 3.6) by an explicit statement that "
      "deviation detection requires the baseline error to be small relative to the deviations sought, which is not the "
      f"case here (RMSE ≈ {te['LR']['RMSE']:.1f}–{te['RF']['RMSE']:.1f} kW vs. a test-period mean load of "
      f"{J['desc']['test_mean']:.2f} kW). The 8.890 kW value was a model residual, not a measured load, and no maintenance "
      "log, alarm record or chiller trend is available to corroborate a fault; the fault attribution has therefore been "
      f"withdrawn (the largest RF residual in the revised analysis is {dev['RF']['MaxDev']:.3f} kW on 12 December 2025, "
      "00:00). The hourly targeting recommendations (00:00, 01:00, 11:00, 18:00, 23:00) have been withdrawn. As a check of "
      "the reviewer's hypothesis, the mean hourly residual with a 95% confidence interval was recomputed after one-hot "
      "encoding (Figure 8): positive residuals remain at 00:00–02:00 and 19:00, so the pattern is not purely an encoding "
      "artefact, but given the error level it is reported only as an indication of higher night-time and evening "
      "consumption in November–December, not as an actionable target.",
      "Sections 3.4, 3.6 and 3.8; Figure 8; Table 5; Conclusions."),
 14: ("We thank the reviewer for this insightful point. A persistence benchmark (Ŷ(t) = Y(t−1)), an RF without Lag-1 and a "
      "calendar-only RF were added (Table 4, Figure 3). Persistence is the worst model in every evaluation (test RMSE = "
      f"{te['Persistence']['RMSE']:.3f} kW). Removing Lag-1 leaves RF test accuracy unchanged ({te['RF_noLag1']['RMSE']:.3f} vs. "
      f"{te['RF']['RMSE']:.3f} kW; DM p = {dm['RF_noLag1_vs_RF'][1]:.2f}), and the calendar-only RF has the lowest test RMSE "
      f"({te['RF_calendar']['RMSE']:.3f} kW, R² = {te['RF_calendar']['R2']:.3f}). Calendar features alone therefore baseline "
      "this building at least as well as the autoregressive models, and the high MDI importance of Lag-1 "
      f"({mdi['Lag-1']:.1f}%) does not translate into out-of-sample accuracy. The discussion now notes that Lag-1 absorbs "
      "anomalies that persist beyond one hour, which is why M&V baselines exclude the dependent variable's own recent "
      "history. The deviation analysis is now computed on the untouched test set, which is not used for model selection.",
      "Section 2.6; Table 4; Figure 3; Sections 3.3, 3.4 and 3.7; Figure 7."),
 15: ("We thank both reviewers. “Significantly” has been removed and the truncated sentence in Section 3.2 has been "
      "completed. All three conclusions have been rewritten and are stated explicitly as conditional on the revised "
      "validation design and on a single building-year: (1) RF was preferred on validation but did not outperform LR on the "
      "untouched test set; (2) the BAS-only feature set is insufficient for an M&V-grade baseline, and Lag-1 adds no "
      "out-of-sample accuracy; (3) residuals above 2 × RMSE identify candidate episodes, not confirmed waste, and "
      "hour-specific recommendations are withheld. The dashboard has been removed as a contribution (see comment 12). As "
      "recommended, the chronological-split justification, the no-tuning disclosure, the limitations section (now with a "
      "fourth limitation) and the LR scatter-slope diagnosis are retained.",
      "Section 3.2; Section 3.9; Conclusions."),
 16: ("We thank the reviewer. We traced the offset to the caption numbering fields: captions are numbered automatically, but "
      "the Figure 1 caption had been typed as plain text, so the automatic numbering started at the time-series plot and "
      "shifted subsequent numbers. The Figure 1 caption now uses the same numbering field; every caption sits directly at "
      "its own table or figure and the numbering is consecutive (Tables 1–5, Figures 1–9). The stray “Table 1” "
      "caption above the Introduction has been removed.",
      "All table and figure captions; Figure 1 caption."),
 17: ("Corrected. Section 3.1 refers to the time-series plot as Figure 2 throughout, and with the caption numbering fixed "
      "(comment 16) the caption of that plot now also reads Figure 2.",
      "Section 3.1; captions of Figures 1 and 2."),
}

zin = zipfile.ZipFile(SRC)
doc = etree.fromstring(zin.read("word/document.xml"))
body = doc.find(q("body")); K = list(body)


def set_text(p, text):
    r0 = next((r for r in p.iter(q("r")) if r.find(q("t")) is not None), None)
    rpr = copy.deepcopy(r0.find(q("rPr"))) if r0 is not None and r0.find(q("rPr")) is not None else None
    for c in [c for c in p if c.tag != q("pPr")]: p.remove(c)
    r = etree.SubElement(p, q("r"))
    if rpr is not None: r.append(rpr)
    t = etree.SubElement(r, q("t")); t.text = text; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def cell_text(tc):
    return "".join(tc.itertext())


tbl = K[5]
for tr in tbl.findall(q("tr"))[1:]:
    tcs = tr.findall(q("tc")); no = int(cell_text(tcs[0]).strip())
    if no not in RESP: continue
    for tc, txt in zip(tcs[4:6], RESP[no]):
        ps = tc.findall(q("p"))
        for extra in ps[1:]: tc.remove(extra)
        set_text(ps[0], txt)
# hapus catatan placeholder di akhir dokumen
for p in K:
    if p.tag == q("p") and "TO BE COMPLETED" in "".join(p.itertext()): body.remove(p)

left = [m for m in re.findall(r"\[[^\]]*\]", "".join(doc.itertext())) if "TO BE" in m or "/" in m]
assert not any("TO BE" in m for m in left), left
zout = zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == "word/document.xml":
        data = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
    zout.writestr(it, data)
zout.close()
print("Tersimpan:", DST, "| sisa [..]:", left)
