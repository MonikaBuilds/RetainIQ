"""RetainIQ - claims register: every headline number, where it comes from, its CI.

Generated from the output files, never typed by hand.
"""
import json

from config import OUT, REPORTS


def ci(d):
    return f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def main():
    E = json.load(open(OUT / "eda_results.json"))
    M = json.load(open(OUT / "metrics.json"))
    P = json.load(open(OUT / "prioritisation_results.json"))
    PL = json.load(open(OUT / "pilot_plan.json"))
    D = json.load(open(OUT / "demo_unseen_results.json"))
    H = E["hypotheses"]
    o = E["overall"]
    hm = M["holdout_metrics"]
    seg = E["table_segment"]
    bt = P["backtest_all_10k"]
    rows = [
        ("DESCRIPTIVE", "Overall churn rate", f"{o['churn_rate']:.2%} ({o['churners']:,} of {o['customers']:,})",
         f"[{o['churn_rate_ci'][0]:.4f}, {o['churn_rate_ci'][1]:.4f}] Wilson", "eda.py -> eda_results.json:overall"),
        ("DESCRIPTIVE", "Share of total balance held by churners", f"{o['churned_balance_share']:.1%}",
         f"winsorised p99: {o['churned_balance_share_winsor99']:.1%}", "eda.py -> overall.churned_balance_share"),
        ("DESCRIPTIVE", "Accuracy of predicting 'nobody churns'", f"{o['accuracy_of_predict_nobody']:.1%}", "-", "eda.py"),
        ("DESCRIPTIVE", "Churn: Mass Market / Mass Affluent / Affluent / Private",
         " / ".join(f"{seg[s]['churn_rate']:.1%}" for s in ("Mass Market", "Mass Affluent", "Affluent", "Private Banking")),
         " / ".join(f"[{seg[s]['ci_low']:.3f},{seg[s]['ci_high']:.3f}]" for s in ("Mass Market", "Private Banking")) + " (MM, PB)",
         "eda.py -> table_segment"),
        ("DESCRIPTIVE", "Mass Market share of churners vs of churned balance",
         f"{E['volume_vs_value']['mass_market_share_of_churners']:.0%} vs {E['volume_vs_value']['mass_market_share_of_churned_balance']:.0%}", "-",
         "eda.py -> volume_vs_value"),
        ("DESCRIPTIVE", "Top balance quintile: share of churners vs churned balance",
         f"{E['volume_vs_value']['top_quintile_balance_share_of_churners']:.0%} vs {E['volume_vs_value']['top_quintile_balance_share_of_churned_balance']:.0%}",
         "-", "eda.py -> volume_vs_value"),
        ("DESCRIPTIVE", "Churn by complaints 0 / 1 / 2 / 3+",
         " / ".join(f"{v:.1%}" for v in H["H1_complaints_dose_response"]["rates"].values()),
         f"trend z={H['H1_complaints_dose_response']['ca_z']:.1f}, BH-p={E['bh_adjusted_p']['H1']:.1e}", "eda.py H1"),
        ("DESCRIPTIVE", "Single- vs multi-product churn",
         f"{H['H3_single_product']['single_rate']:.1%} vs {H['H3_single_product']['multi_rate']:.1%}",
         f"OR {H['H3_single_product']['odds_ratio'][0]:.2f} [{H['H3_single_product']['odds_ratio'][1]:.2f}, {H['H3_single_product']['odds_ratio'][2]:.2f}]", "eda.py H3"),
        ("DESCRIPTIVE", "Churn by days inactive <=15 / 16-45 / 46-90 / 90+",
         " / ".join(f"{v:.1%}" for v in H["H4_inactivity"]["rates"].values()),
         f"trend z={H['H4_inactivity']['ca_z']:.1f}", "eda.py H4"),
        ("DESCRIPTIVE", "Churn by tenure <1 / 1-3 / 3-7 / 7-15 / 15+",
         " / ".join(f"{v:.1%}" for v in H["H6_early_tenure"]["rates"].values()),
         f"trend z={H['H6_early_tenure']['ca_z']:.1f}", "eda.py H6"),
        ("DESCRIPTIVE", "RM odds ratio crude vs adjusted (segment, balance, tenure, products)",
         f"{H['H7_rm_confounding']['crude_or'][0]:.2f} vs {H['H7_rm_confounding']['adjusted_or'][0]:.2f}",
         f"adj. [{H['H7_rm_confounding']['adjusted_or'][1]:.2f}, {H['H7_rm_confounding']['adjusted_or'][2]:.2f}]", "eda.py H7"),
        ("DESCRIPTIVE", "Region / gender association with churn (Cramer's V)",
         f"{H['H9_region_gender_flat']['region']['cramers_v']:.3f} / {H['H9_region_gender_flat']['gender']['cramers_v']:.3f}",
         f"p={H['H9_region_gender_flat']['region']['p']:.2f} / {H['H9_region_gender_flat']['gender']['p']:.2f}", "eda.py H9"),
        ("DESCRIPTIVE", "Clustering: k, bootstrap stability (ARI)",
         f"k={E['clustering']['k']}, ARI {E['clustering']['bootstrap_ari_mean']:.2f}", f"sd {E['clustering']['bootstrap_ari_sd']:.3f}", "eda.py clustering"),
        ("PREDICTIVE", "Deployed model / variant", f"{M['model_selection'][M['deployment']['variant']]['chosen']} / {M['deployment']['variant']}",
         "-", "train.py deployment rule"),
        ("PREDICTIVE", "Leakage test ROC-AUC A (all) / B (no inactivity) / C (no inactivity, no NPS)",
         " / ".join(f"{M['leakage_experiment_holdout'][v]['roc_auc']:.3f}" for v in ("A_full", "B_no_inactivity", "C_early_warning")),
         "hold-out, n=2,000", "train.py leakage_experiment_holdout"),
        ("PREDICTIVE", "Hold-out ROC-AUC", f"{hm['roc_auc']['value']:.3f}", ci(hm["roc_auc"]) + " bootstrap", "train.py holdout_metrics"),
        ("PREDICTIVE", "Hold-out PR-AUC (base rate 0.158)", f"{hm['pr_auc']['value']:.3f}", ci(hm["pr_auc"]), "train.py"),
        ("PREDICTIVE", "Top-10% lift / recall", f"{hm['lift_top10']['value']:.2f}x / {hm['recall_top10']['value']:.1%}",
         ci(hm["lift_top10"]) + " (lift)", "train.py"),
        ("PREDICTIVE", "Rule-of-thumb baseline ROC-AUC / top-10% lift",
         f"{M['holdout_baselines']['rule_of_thumb']['roc_auc']:.3f} / {M['holdout_baselines']['rule_of_thumb']['lift_top10']:.2f}x", "-", "train.py"),
        ("PREDICTIVE", "Stability over 10 random splits (ROC-AUC)", f"{M['stability_10_seeds']['roc_mean']:.3f} +/- {M['stability_10_seeds']['roc_sd']:.3f}",
         "mean +/- sd", "train.py stability_10_seeds"),
        ("PREDICTIVE", "Brier score (calibrated)", f"{hm['brier']['value']:.4f}", ci(hm["brier"]), "train.py"),
        ("PREDICTIVE", "Augmentation ablation ROC-AUC real 8k vs real+synthetic 48k (logreg)",
         f"{M['augmentation']['ablation_holdout']['logreg|real_8k']['roc_auc']['value']:.3f} vs {M['augmentation']['ablation_holdout']['logreg|real+synthetic_48k']['roc_auc']['value']:.3f}",
         "no gain", "train.py augmentation"),
        ("PREDICTIVE", "Real-vs-synthetic discriminator AUC", f"{M['augmentation']['validation']['real_vs_synthetic_auc']:.3f}",
         "0.5 = indistinguishable", "augment.py validate"),
        ("PREDICTIVE", "Unseen-data demo (train-split model, labels removed) ROC-AUC / top-decile hit rate",
         f"{D['clean_unlabelled']['roc_auc']:.3f} / {D['clean_unlabelled']['top_decile_hit_rate']:.1%}",
         f"dirty file: {D['dirty_unlabelled']['roc_auc']:.3f}", "demo_unseen.py"),
        ("RECOMMENDATION", "Back-test: churned balance caught with 500 contacts, RetainIQ vs risk-only vs balance-only",
         f"{bt['Probability x balance (RetainIQ)']['500']['share_churned_balance']:.0%} vs {bt['Probability only']['500']['share_churned_balance']:.0%} vs {bt['Balance only']['500']['share_churned_balance']:.0%}",
         "OOF probabilities, actual outcomes", "prioritise.py backtest_all_10k"),
        ("RECOMMENDATION", "Active customers ranked", f"{P['active_customers']:,}", "-", "prioritise.py"),
        ("RECOMMENDATION", "P1 size and share of balance at risk",
         f"{P['tier_summary']['P1']['customers']:.0f} customers, {P['tier_summary']['P1']['share_balance_at_risk']:.0%}", "-", "prioritise.py tier_summary"),
        ("RECOMMENDATION", "Pilot: population, per arm, detectable effect",
         f"P1+P3, {min(v['n'] for v in PL['arms'].values())} per arm, MDE {PL['achievable_mde_pp']:.1%} points",
         "alpha 0.05, power 0.80", "pilot_power.py"),
    ]
    lines = ["# RetainIQ claims register", "",
             "Every headline number in the report, the script that produced it and its uncertainty. "
             "Generated by `src/claims_register.py` from the output files.", "",
             "| Type | Claim | Value | Uncertainty | Source |", "|---|---|---|---|---|"]
    lines += [f"| {t} | {c} | {v} | {u} | `{s}` |" for t, c, v, u, s in rows]
    (REPORTS / "claims_register.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
