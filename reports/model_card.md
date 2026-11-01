# Model card: RetainIQ churn-risk model

| Item | Detail |
|---|---|
| Model | L2-regularised logistic regression (C = 0.1), median imputation + standardisation, Platt (sigmoid) calibration (5-fold). File: `outputs/retainiq_model.pkl` |
| Deployed variant | **B: no `last_transaction_days_ago`**. Inactivity is used as a live trigger in the playbook, not as a model input |
| Purpose | Rank *active* retail customers by similarity to past leavers; combined with balance to set retention priorities |
| Intended use | Prioritising human-reviewed, opt-out-able retention outreach; sizing a randomised pilot |
| Not intended for | Credit, pricing or eligibility decisions; automated adverse action; forecasting churn over a fixed time window; judging individual RM performance |
| Training data | 10,000 simulated customers (`Bank_Data.csv`), cleaned by `src/clean.py`. Hold-out of 2,000 never used for fitting, tuning, calibration or thresholds |
| Target | `churned` (1 = left). Historical snapshot; no date, so no horizon |
| Features (21) | Segment rank, tenure, log income, log balance, balance/income, number of products, single-product flag, card / loan / investment flags, products per tenure year, digital engagement, low-engagement flag, branch visits, complaints (capped at 3+), credit score, RM assigned, high balance without RM, two data-quality flags, NPS |
| Excluded | `customer_id`; **gender, age, region** (audit only); `Aging` (redundant); `product_flag_count` (exact sum of the flags) |

## Performance (hold-out, n = 2,000, 95% bootstrap CI)

| Metric | Value |
|---|---|
| ROC-AUC | 0.821 [0.796, 0.843] (10 random splits: 0.808 ± 0.010) |
| PR-AUC (base 0.158) | 0.499 [0.444, 0.554] |
| Lift / recall in top 10% | 3.45× / 34.5% |
| Brier | 0.104 [0.096, 0.114] |
| At operating flag (top 25% risk, t = 0.22) | precision 0.40, recall 0.67, F1 0.50 |

Baselines: predicting the base rate gives AUC 0.50; a complaints + inactivity rule gives AUC 0.663 and 2.34× lift.

## Explainability

- **Global:** permutation importance, standardised odds ratios, a SHAP view from a LightGBM comparator, and partial dependence plots (`outputs/figures/14`–`16`).
- **Local:** three plain-English reason codes per customer from per-feature log-odds contributions, e.g. "single product + 2 complaint(s) last year + low digital engagement (33/100)".
- **Correlated features:** segment, balance, income, RM and products form a correlated wealth block (VIF up to ~13). Read their weights as a group, not as independent drivers.

## Fairness audit (hold-out, operating flag)

- **Gender:** AUC gap 0.018; recall gap 0.06.
- **Calibration by group:** mean prediction is within 0.01 of the actual rate in every segment.
- **Recall by segment** ranges from 0.14 (Affluent) to 0.78 (Mass Market). This reflects base rates (5% vs 22%); value-based tiering routes low-rate, high-balance customers into P2 instead.
- **Tier mix by gender** is near-identical.
- `[LIMITATION]` Tenure and products may proxy for age; no explicit age effect is used.

## Robustness

| Test | Result |
|---|---|
| 5% noise | AUC 0.819 |
| 10% missing cells | AUC 0.807 |
| Top feature (single_product) removed | AUC 0.791 |
| Out-of-range values | Handled by cleaning |
| Base-rate shift to 8% or 30% | Ranking AUC holds (0.82–0.83), but mean prediction stays ~15–19%, so **recalibrate when the base rate moves** |
| Train vs hold-out drift | Max PSI 0.006 |

## Monitoring and retraining

Monthly: AUC and PR-AUC on realised outcomes, PSI per feature, calibration-in-the-large.

Retrain when any of these holds:
- AUC < 0.78
- any feature PSI > 0.2
- predicted vs observed churn gap > 3 percentage points
- 12 months have passed

Human-in-the-loop: the model recommends, an RM decides, and any customer can be suppressed from outreach.
