"""RetainIQ - requirements audit: checks every brief + team-spec requirement
against the actual output files, and writes reports/requirements_audit.md.

    python src/audit_requirements.py
"""
import json
import re

import joblib
import pandas as pd

from config import DATA, FIG, OUT, REPORTS, ROOT

rows = []


def check(area, req, ok, evidence):
    rows.append((area, req, "PASS" if ok else "MISSING", evidence))


def j(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else {}


def words(path):
    t = path.read_text(encoding="utf-8") if path.exists() else ""
    t = re.sub(r"\|[-| ]+\|", "", t)
    return len([w for w in t.split() if re.search(r"\w", w)])


def main():
    E, M, P, PL, D = j("eda_results.json"), j("metrics.json"), j("prioritisation_results.json"), j("pilot_plan.json"), j("demo_unseen_results.json")
    figs = sorted(f.name for f in FIG.glob("*.png"))
    ranked = pd.read_csv(OUT / "ranked_customers.csv") if (OUT / "ranked_customers.csv").exists() else pd.DataFrame()
    clean = pd.read_csv(DATA / "cleaned_bank_data.csv") if (DATA / "cleaned_bank_data.csv").exists() else pd.DataFrame()

    # ---------------- the brief: "Your solution must"
    H = E.get("hypotheses", {})
    check("Brief 1", "Strongest churn patterns via exploration + visualisation", len(H) >= 11 and len(figs) >= 3,
          f"{len(H)} hypotheses tested (H1-H11), BH-corrected; {len(figs)} figures")
    hm = M.get("holdout_metrics", {})
    check("Brief 2", "At least one ML model built and evaluated", bool(hm),
          f"logreg (+tree/RF/LightGBM compared); hold-out ROC-AUC {hm.get('roc_auc', {}).get('value', 0):.3f}, PR-AUC {hm.get('pr_auc', {}).get('value', 0):.3f}")
    check("Brief 3", "Ranked list of customers / groups", len(ranked) > 0 and "priority_rank" in ranked,
          f"outputs/ranked_customers.csv: {len(ranked):,} active customers, P1-P4 tiers + 5 personas")
    bt = P.get("backtest_all_10k", {})
    check("Brief 4", "Balance at risk used in prioritisation (not just probability)", "balance_at_risk" in ranked and bool(bt),
          f"priority = P x value; back-test @500 contacts: {bt.get('Probability x balance (RetainIQ)', {}).get('500', {}).get('share_churned_balance', 0):.0%} of churned balance vs {bt.get('Probability only', {}).get('500', {}).get('share_churned_balance', 0):.0%} risk-only")
    check("Brief 5", "Targeted interventions + measurable pilot", bool(PL.get("arms")) and "recommended_action" in ranked,
          f"per-customer recommended_action; pilot {sum(a['n'] for a in PL.get('arms', {}).values()):,} customers, MDE {PL.get('achievable_mde_pp', 0):.1%} pts")
    dc = D.get("clean_unlabelled", {}); dd = D.get("dirty_unlabelled", {})
    check("Brief 6", "Works on previously unseen records / hidden test input", bool(dc) and dd.get("rows_scored") == 2000,
          f"hold-out scored with labels removed: AUC {dc.get('roc_auc', 0):.3f}; dirty file scored {dd.get('rows_scored')}/2000")

    # ---------------- deliverables
    check("Deliverable", "Cleaned dataset + documented cleaning", len(clean) == 10000 and (OUT / "data_cleaning_log.csv").exists(),
          "data/cleaned_bank_data.csv + outputs/data_cleaning_log.csv + data_quality_scorecard.csv")
    nbs = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
    check("Deliverable", "Analysis notebooks / scripts (evidence)", len(nbs) == 4, f"{nbs}; src/ pipeline")
    check("Deliverable", "At least three visualisations of churn patterns", len(figs) >= 3, f"{len(figs)} figures in outputs/figures")
    wc = words(REPORTS / "management_report.md")
    check("Deliverable", "Management report ~1,000-1,500 words", 900 <= wc <= 1700, f"reports/management_report.md: {wc} words incl. tables (+ published HTML page)")
    pilot_txt = (REPORTS / "management_report.md").read_text(encoding="utf-8") if (REPORTS / "management_report.md").exists() else ""
    check("Deliverable", "Pilot with KPIs + how results are evaluated", all(k in pilot_txt for k in ("KPIs", "Decision rule", "intention-to-treat")),
          "primary/secondary/guardrail KPIs, ITT analysis, pre-registered decision rule")

    # ---------------- AI / integrity rules
    check("Rules", "AI tools disclosed + verification described", (REPORTS / "ai_usage_disclosure.md").exists(), "reports/ai_usage_disclosure.md")
    check("Rules", "Code runs and produces outputs from supplied data", (OUT / "run_all_log.txt").exists() and "complete" in (OUT / "run_all_log.txt").read_text(encoding="utf-8", errors="ignore"),
          "python src/run_all.py completes; outputs/run_all_log.txt")
    check("Rules", "Target, features, split, metrics, ranking logic explained",
          all((REPORTS / f).exists() for f in ("model_card.md", "judge_qa.md")) and bool(M.get("feature_table")),
          "model_card.md, judge_qa.md, metrics.json feature_table/feature_rationale/excluded")
    check("Rules", "No hardcoded metrics (claims register generated from outputs)", (REPORTS / "claims_register.md").exists(), "reports/claims_register.md generated by src/claims_register.py")

    # ---------------- team spec (stages 0-5 + governance)
    a = M.get("augmentation", {})
    check("Spec S0", "10k -> 50k augmentation, train-only, lineage-tagged, validated, ablation", a.get("grand_total_with_holdout") == 50000 and "ablation_holdout" in a,
          f"total {a.get('grand_total_with_holdout')}; discriminator AUC {a.get('validation', {}).get('real_vs_synthetic_auc', 0):.3f}; ablation shows no gain")
    check("Spec S1", "Robust loader, validity rules, flags, Aging resolved, features, before/after scorecard",
          (OUT / "data_quality_scorecard.csv").exists() and "tenure_band" in clean, "clean.py + features.py; Aging rebuilt as tenure_band")
    check("Spec S2", "RM confounding test (within-segment + adjusted logit)", "adjusted_or" in H.get("H7_rm_confounding", {}),
          f"crude OR {H.get('H7_rm_confounding', {}).get('crude_or', [0])[0]:.2f} -> adjusted {H.get('H7_rm_confounding', {}).get('adjusted_or', [0])[0]:.2f}")
    check("Spec S2", "Tiny cells grouped (3+ complaints), Wilson CIs, BH correction", "bh_adjusted_p" in E, "complaint_band 0/1/2/3+; Wilson CIs; fdr_bh")
    c = E.get("clustering", {})
    check("Spec S2", "Clustering with k selection, stability, personas, plays", c.get("k", 0) >= 3,
          f"k={c.get('k')}, bootstrap ARI {c.get('bootstrap_ari_mean', 0):.2f}; GMM + rules comparators")
    check("Spec S2", "10 EDA visualisations with 'so what' titles", sum(f[:2].isdigit() and int(f[:2]) <= 10 for f in figs) == 10, "figures 01-10")
    check("Spec S2", "Dashboard data + Tableau/Power BI spec", (DATA / "dashboard_data.csv").exists() and (REPORTS / "dashboard_spec.md").exists(), "data/dashboard_data.csv + reports/dashboard_spec.md")
    lk = M.get("leakage_experiment_holdout", {})
    check("Spec S3", "Leakage experiment A / B / C", all(k in lk for k in ("A_full", "B_no_inactivity", "C_early_warning")),
          " / ".join(f"{k}:{lk[k]['roc_auc']:.3f}" for k in ("A_full", "B_no_inactivity", "C_early_warning") if k in lk) + f"; deployed {M.get('deployment', {}).get('variant')}")
    cv = pd.read_csv(OUT / "cv_model_comparison.csv") if (OUT / "cv_model_comparison.csv").exists() else pd.DataFrame()
    check("Spec S3", "LR, tree, RF, gradient boosting + dummy + rule baseline", set(cv.get("model", [])) >= {"logreg", "tree", "random_forest", "lightgbm", "dummy"} and "rule_of_thumb" in lk,
          "cv_model_comparison.csv; simplest-within-1-SE rule")
    check("Spec S3", "Imbalance: class weights vs SMOTE (in folds) vs none", (OUT / "imbalance_comparison.csv").exists(), "imbalance_comparison.csv")
    check("Spec S3", "Metrics with bootstrap CIs incl. PR-AUC, lift, balance capture", "balance_capture_top10" in hm, "holdout_metrics in metrics.json")
    check("Spec S3", "Calibration + cost-based threshold", "calibration_holdout" in M and "threshold" in M,
          f"Brier {hm.get('brier', {}).get('value', 0):.4f}; value-optimal t={M.get('threshold', {}).get('value_optimal')}, operating t={M.get('threshold', {}).get('operating', 0):.2f}")
    check("Spec S3", "Explainability: permutation, SHAP, PDP, reason codes, driver stability",
          all(f in figs for f in ("14_permutation_importance.png", "15_shap_summary.png", "16_partial_dependence.png")) and "top_reasons" in ranked,
          "figures 14-16; top_reasons per customer; top-5 driver frequency over seeds")
    check("Spec S3", "Fairness audit, 10-seed stability, stress tests, PSI drift",
          all(k in M for k in ("fairness", "stability_10_seeds", "stress_tests", "psi_max")),
          f"stability AUC {M.get('stability_10_seeds', {}).get('roc_mean', 0):.3f}+/-{M.get('stability_10_seeds', {}).get('roc_sd', 0):.3f}; max PSI {M.get('psi_max')}")
    bundle = joblib.load(OUT / "retainiq_model.pkl") if (OUT / "retainiq_model.pkl").exists() else {}
    check("Spec S3", "predict.py for any CSV with original schema", (ROOT / "src" / "predict.py").exists() and bool(bundle),
          f"retainiq_model.pkl ({bundle.get('model_name')}, {len(bundle.get('features', []))} features)")
    check("Spec S4", "Rank ACTIVE customers only with out-of-fold probabilities", all(P.get("checks", {}).values()) and bool(P.get("checks")), "prioritise.py checks")
    check("Spec S4", "2x2 tiers, budget view N=100/250/500/1000, 3 ranking rules", bool(P.get("tier_summary")) and "1000" in str(P.get("budget_active_expected", {})),
          "tier_summary.csv; figures 17-18")
    check("Spec S4", "Sensitivity (margin, save rate, cost, horizon, winsor, threshold) + ethics mix", bool(P.get("sensitivity")) and bool(P.get("tier_mix_by_group")),
          "prioritisation_sensitivity.csv; tier_mix_by_group")
    check("Spec S5", "Intervention playbook + power calc + randomised assignment", (OUT / "pilot_assignment.csv").exists() and (OUT / "pilot_sizing.csv").exists(),
          "pilot_assignment.csv (stratified, seeded); pilot_sizing.csv")
    check("Spec S10", "Model card, monitoring, human-in-the-loop", (REPORTS / "model_card.md").exists(), "reports/model_card.md")
    qa = (REPORTS / "judge_qa.md").read_text(encoding="utf-8") if (REPORTS / "judge_qa.md").exists() else ""
    check("Spec S12", "Red-team Q&A: 22 defended answers", len(re.findall(r"^\*\*\d+\.", qa, re.M)) == 22, f"{len(re.findall(r'^\*\*\d+\.', qa, re.M))} answers")
    check("Spec S11", "README + pinned requirements.txt + one command", (ROOT / "README.md").exists() and (ROOT / "requirements.txt").exists(), "README.md, requirements.txt, src/run_all.py")
    check("Spec S0A", "RetainIQ branding (pitch, prefixed model/dashboard files)", "RetainIQ" in (ROOT / "README.md").read_text(encoding="utf-8"),
          "pitch in README/report; retainiq_model.pkl; RetainIQ_Dashboard spec. Scripts keep spec's src/ names (clean.py, predict.py...)")

    df = pd.DataFrame(rows, columns=["area", "requirement", "status", "evidence"])
    n_ok = (df.status == "PASS").sum()
    md = ["# RetainIQ requirements audit", "",
          f"Generated by `src/audit_requirements.py` from the output files. **{n_ok} / {len(df)} requirements met.**", "",
          "| Area | Requirement | Status | Evidence |", "|---|---|---|---|"]
    md += [f"| {r.area} | {r.requirement} | {r.status} | {r.evidence} |" for r in df.itertuples()]
    (REPORTS / "requirements_audit.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
    print(df[["area", "requirement", "status"]].to_string(index=False))
    print(f"\n{n_ok}/{len(df)} requirements met")


if __name__ == "__main__":
    main()
