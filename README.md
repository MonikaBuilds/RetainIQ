# RetainIQ: Predict. Prioritise. Retain.

> *RetainIQ is the AI that tells Meridian Trust Bank which customers are about to leave, and which of them are worth saving.*

Churners are **15.8%** of customers but hold only **6.6%** of balances. RetainIQ stops asking "who will leave?" and answers "who is **worth saving**?"

## Reproduce everything (one command)

```bash
pip install -r requirements.txt        # Python 3.13, pinned versions
python src/run_all.py                  # raw CSV -> every table, figure, model, report number (~100 s)
python src/make_notebooks.py           # optional: rebuild the 4 executed evidence notebooks
```

Fixed seed (42). The run prints PASS/FAIL assertions at every stage.

## Interactive dashboard

```bash
streamlit run app/retainiq_app.py      # opens http://localhost:8501
```

The dashboard has six tabs:
- **Overview**: KPIs and the thesis.
- **See**: churn drivers with 95% CIs.
- **Predict**: validation, leakage test, calibration, fairness.
- **Prioritise**: WorthSaving matrix, budget simulator, priority list with CSV export, customer card.
- **Act**: playbook and pilot.
- **Score a file**: upload any CSV, dirty files included.

Sidebar filters cover segment, region, tenure, RM and persona. Assumption sliders for margin, save rate, horizon and contact cost re-rank the list live. Every number is read from pipeline outputs, so run `src/run_all.py` first. The design rationale is in `reports/ux_research_brief.md`.

## Score new customers (unseen / hidden test file)

```bash
python src/predict.py --input path/to/new_customers.csv --output outputs/scored_customers.csv
```

The input can be dirty: junk header rows, pasted totals, numbers as text, any column order, missing or out-of-range values.

Output columns: `customer_id, churn_probability, risk_tier, priority_tier, balance_at_risk, net_retention_value, priority_score, priority_rank, top_reasons, recommended_action`.

A live proof is in `src/demo_unseen.py`: the hold-out is scored with labels removed by a model that never saw it, then compared to the truth.

## Story arc

| Stage | Script | Headline |
|---|---|---|
| **See**: clean | `src/clean.py` | 18 pasted summary rows, a junk header and 9,895 text-numbers removed. 10,000 clean customers; logic issues flagged, never silently dropped |
| **See**: EDA | `src/eda.py` | Single product (36% churn), complaints (dose-response), early tenure and inactivity. The RM effect is mostly wealth confounding |
| **Predict** | `src/train.py` | Logistic regression (simplest within 1 SE): hold-out AUC 0.821, 3.45× top-decile lift. Leakage-tested; augmentation to 50k honestly reported as no gain |
| **Prioritise** | `src/prioritise.py` | Risk × balance catches 36% of leaving money with 500 contacts, vs 9% for risk only |
| **Act** | `src/pilot_power.py` | Randomised pilot: 2 × ~1,050 high-risk customers, detects a 5.5-point drop |

## Repository map

```
data/raw/Bank_Data.csv          original file (untouched)
data/cleaned_bank_data.csv      cleaned + engineered features
data/augmented_train.csv        8k real train + 40k synthetic (lineage-tagged)
data/dashboard_data.csv         one row per customer for Tableau / Power BI
data/demo/                      unlabelled + deliberately dirty hold-out files and their scores
src/                            pipeline (config, clean, features, eda, augment, modeling, train,
                                prioritise, pilot_power, predict, demo_unseen, claims_register, run_all)
notebooks/                      01_cleaning, 02_eda_clustering, 03_model, 04_prioritisation (executed)
outputs/figures/                18 figures
outputs/                        ranked_customers.csv, retainiq_model.pkl, metrics.json, data_cleaning_log.csv,
                                fairness_audit.csv, pilot_assignment.csv, ...
reports/                        management_report.md, model_card.md, judge_qa.md, claims_register.md,
                                ai_usage_disclosure.md, dashboard_spec.md
```

## Key assumptions (all in `src/config.py`, all sensitivity-tested)

`[ASSUMPTION]` These are judgement calls, not facts from the data:

- **Value of a saved customer** = (balance × 2.5% margin + 50 per product) × a 3-year horizon. Balance is a stock, so it is **not** multiplied by 12.
- **Save rate:** 20%.
- **Contact costs:** 150 (RM call), 40 (nurture), 5 (digital), 0 (monitor).
- **Protected attributes:** gender, age and region are used for auditing only, never as model inputs.

## Limitations

`[LIMITATION]` These are known weaknesses in the data:

- No dates, so there is no forecast horizon.
- No competitor field, so "switching" is a risk profile, not confirmed switching.
- No treatment history, so the model predicts *risk*, not *persuadability*. The randomised pilot measures uplift.
