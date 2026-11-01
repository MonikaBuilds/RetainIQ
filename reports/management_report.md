# RetainIQ: Predict. Prioritise. Retain.

**Report to the Meridian Trust Bank Management Committee**

> *RetainIQ is the AI that tells Meridian Trust Bank which customers are about to leave, and which of them are worth saving.*

Every number below is reproduced by `python src/run_all.py` from the raw file and traced in `reports/claims_register.md`. Labels: **[D]** descriptive (what happened), **[P]** predictive (model estimate), **[R]** recommendation (our judgement). `[ASSUMPTION]` marks a judgement call and `[LIMITATION]` marks a known weakness.

---

## 1. The problem: churn is a volume problem, but the money is elsewhere

**[D]** 1,578 of 10,000 customers churned: **15.8%** (95% CI 15.1–16.5%). Those churners held only **6.6%** of total balances. Three in four churners (73%) are Mass Market, but they account for just 19% of the balance that left. The top balance quintile is only 7% of churners yet holds **56% of the churned balance** (Figure 3).

**[R]** "Who will leave?" is therefore the wrong question. The right one is **"who is worth saving?"** A list of the most-likely leavers is mostly low-balance customers. Spending retention budget on them protects headcount, not deposits.

## 2. Data quality: the problems were structural, not statistical

**[D]** The raw file had a junk title row, 18 pasted summary rows ("Total", "Churn %") that would have created a fake balance of 342 million, an undocumented `Aging` column, an empty column, and 9,895 numbers stored as text ("1,18,776"). Once these were removed there were **10,000 customers, no missing values and no duplicates**.

Logic issues were flagged, not deleted: 199 rows where product flags exceed `num_products`; 1,009 where tenure exceeds adult life (38 exceed age); 157 incomes floored at 1,500; 1,870 credit scores capped at 850. All 137 `Aging` mismatches sit exactly on a band edge (1/3/7/15 years): the source mixed inclusive and exclusive boundaries.

The loader auto-detects the header and handles text numbers, any column order and missing values. A deliberately corrupted test file scored all 2,000 customers correctly.

## 3. What is associated with churn

**[D]** All tests are on the 10,000 real customers. P-values are Benjamini-Hochberg corrected. These are associations, not causes.

| Signal | Churn rate | Note |
|---|---|---|
| Products held: 1 vs 2+ | **36.1% vs 8.8%** | Odds ratio 5.9 [5.2–6.6], the strongest single signal |
| Complaints: 0 / 1 / 2 / 3+ | 13.1% / 20.1% / 31.1% / 29.9% | Dose-response (z = 12.5); 3+ grouped because n = 67 |
| Days since last transaction: ≤15 / 16–45 / 46–90 / 90+ | 10.6% / 15.1% / 27.0% / **48.7%** | "Silent Exits": possibly a symptom of leaving (§4) |
| Tenure: under 1 year vs 15+ years | 28.1% vs 5.0% | Steady decline with tenure |
| Segment: Mass Market vs Private Banking | 21.8% vs 4.1% | |
| Low digital engagement (bottom quartile) | 26.7% vs 12.1% | |
| Region, gender | Flat | Cramer's V 0.02 and 0.001; **no real difference** |

Single-product customers are the hot spot in *both* mass segments: 35% churn in Mass Market and 44% in Mass Affluent (Figure 8).

**Relationship managers.** RM customers churn less in every segment. But RMs are assigned to 81% of Affluent and Private customers and only 6% of Mass Market, and the median balance is 61k with an RM against 4.9k without. Adjusting for segment, balance, tenure and products moves the odds ratio from 0.24 to **0.38** [0.30–0.50]. A residual association remains, but we **cannot claim RMs cause retention** (Figure 7).

**Personas.** Clustering without using churn found five stable groups (bootstrap ARI 0.92). Two Mass Market personas carry most of the churn:
- **Low-Digital Branch-Reliant**: 28% churn, 25% of the churned balance.
- **Disgruntled Unhappy**: 25% churn, driven by complaints.

The **Multi-Product Wealthy** persona churns at only 4% but holds 44% of the churned balance.

## 4. The model: simple, honest, validated

**[P]** The target is `churned` (1 = left), a historical snapshot with no date. `[LIMITATION]` The model finds customers who *resemble past leavers*; it is not a forecast over a fixed horizon.

We used a stratified 80/20 split with a fixed seed. The 2,000-customer hold-out was never augmented, tuned on, or used for calibration or threshold choice. We compared logistic regression, a decision tree, random forest and LightGBM using repeated 5-fold cross-validation. **Logistic regression was within one standard error of the best model, so we chose it** because it is the most explainable. Gender, age and region are excluded from the model and used only in the fairness audit.

**The leakage test.** Days-inactive and NPS may be *consequences* of leaving rather than early signs. Hold-out ROC-AUC for each variant:

- **A**, all features: 0.845
- **B**, without days-inactive: 0.821
- **C**, without days-inactive and NPS: 0.812

By the rule we set *before* seeing results, we deploy **B**. Inactivity stays in the playbook as a live "Silent Exits" trigger rather than as a model input. B still finds people *before* they go quiet.

**Hold-out performance (variant B, 95% bootstrap CIs):**

| Measure | Value |
|---|---|
| ROC-AUC | **0.821** [0.796–0.843]. Across 10 random splits: 0.808 ± 0.010 |
| PR-AUC (random = 0.158) | 0.499 [0.444–0.554] |
| Top 10% by risk | **3.45× lift**, captures 34.5% of churners. A complaints + inactivity rule of thumb gets 2.34× |
| Calibration | Brier 0.104; mean prediction 16.3% against actual 15.8% (Figure 11) |

Accuracy is deliberately not a headline: predicting "nobody leaves" scores 84.2%.

Main drivers: single product, low digital engagement, complaints and low NPS, the same top four in 5 of 5 resamples (Figure 14).

**Robustness.** 5% noise and 10% missing cells barely move AUC (0.819, 0.807). Train/hold-out drift is negligible (max PSI 0.006). AUC differs by only 0.02 between genders and calibration holds within every segment; recall varies by segment because base rates do, and value-based tiering covers that gap (§5).

**Unseen data.** A model trained without the hold-out scored it with labels removed: ROC-AUC 0.821, and **54.5% of the top decile actually churned** (3.4× the base rate).

**Expanding to 50,000 records.** We generated 40,000 synthetic training rows by interpolating between pairs of real training customers, never hold-out ones. The synthetic rows passed validation: a classifier could barely tell them from real rows (AUC 0.55), and the churn rate and correlations were preserved. They did **not** improve the model (hold-out AUC 0.821 real-only against 0.819 augmented). We report that honestly and deploy the real-data model.

## 5. Who is worth saving

**[R]** We rank only the **8,422 active customers**, using out-of-fold probabilities (each customer is scored by a model that never saw them). For each customer:

- **Balance at risk** = P(churn) × balance.
- **Net retention value** = P(churn) × annual value × 3-year horizon × save rate − intervention cost.

`[ASSUMPTION]` Annual value = balance × 2.5% net interest margin + 50 per product. Save rate is 20%. Contact costs are 150 (RM call), 40 (nurture), 5 (digital) and 0 (monitor). We do not multiply a *balance* by 12: it is a stock, not a monthly flow.

**The decisive evidence** is a back-test on all 10,000 customers using actual outcomes. With 500 contacts:

| Ranking rule | Churned balance caught | Churners caught |
|---|---|---|
| **RetainIQ (risk × balance)** | **36%** | 96 |
| Risk only | 9% | 317 |
| Balance only | 28% | 21 |

RetainIQ trades headcount for value, which is exactly the trade the brief asks for (Figure 17).

**The WorthSaving matrix (Figure 18):**

| | High value (top 25% balance) | Lower value |
|---|---|---|
| **High risk (top 25%)** | **P1 Save now**: 118 customers, 27% risk, 12% of balance at risk | **P3 Low-cost automation**: 1,988 customers, 32% risk |
| **Lower risk** | **P2 Nurture & deepen**: 1,988 customers, 51% of balance at risk | **P4 Monitor**: 4,328 customers, no spend |

**If the team can call only 200 customers:** take the top 200 by net value. Together they carry **26% of all active balance at risk**, with an expected net value of about 62k under our assumptions.

## 6. What to do, and how to prove it

**[R]** Each intervention is matched to the evidence behind it:

| Driver | Action | Cost |
|---|---|---|
| Single product (36% churn) | Second-product bundle + 6-month fee waiver | Low |
| Complaint in the last year | 48-hour resolution SLA + follow-up call | Medium |
| 46+ days inactive | Re-engagement journey (app nudge + first-transaction cashback) | Low |
| High balance, no RM | Assign an RM or digital concierge | High |
| Under 1 year tenure (28% churn) | 90-day onboarding programme | Low |
| High balance, low digital engagement | Assisted digital set-up in branch | Medium |

Low-balance customers are not abandoned: they receive automated, low-cost care (P3). Tier mix is near-identical by gender (P1: 1.5% of women vs 1.3% of men).

**Pilot design:**
- **Population and arms:** 2,106 high-risk customers (P1 + P3), randomised within tier × segment × risk decile (fixed seed): 1,055 treatment vs 1,051 control (no special action). P2's ~4% baseline is too low to measure in a pilot.
- **Power:** detects a **5.5-point** drop (a 20% relative reduction from ~31%) at α 0.05, 80% power, over a pre-registered 120 days.
- **KPIs:** primary is the churn-rate difference (intention-to-treat, with CI). Secondary: balance retained, cost per saved customer, ROI, product depth, NPS, complaint resolution time. Guardrails: complaint rate, opt-outs, margin given away, contact fatigue.
- **Decision rule:** scale if uplift ≥ 4 points, ROI > 1 and guardrails clean; iterate if uplift is positive but ROI < 1; otherwise stop.

The control arm matters because the model identifies who is *likely to leave*, not who is *persuadable*. The pilot measures the true save rate and replaces our 20% assumption.

## 7. Risks and limitations

- **The value assumptions drive the list.** A 1-year instead of 3-year horizon changes 23% of the top-250 list; halving the margin changes 17%; winsorising balances changes 2%. The pilot will measure the real values.
- **No time dimension, no competitor data.** "Switching" is a risk profile, not confirmed switching.
- **Synthetic data does not help.** Only real records are used for reporting and deployment.
- **Calibration under drift.** Calibration fits this base rate only. When churn shifts, the ranking holds but probabilities drift, so we recalibrate monthly.

**Monitoring:** monthly AUC on realised outcomes, PSI on each feature, and a calibration check. Retrain if AUC falls below 0.78 or any feature's PSI exceeds 0.2.

**Governance:** the model recommends and a person decides. Customers can be removed from outreach at any time.
