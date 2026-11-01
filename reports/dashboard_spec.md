# RetainIQ_Dashboard: Tableau / Power BI spec

**Source:** `data/dashboard_data.csv`, one row per customer (all 10,000). Key fields:

| Group | Fields |
|---|---|
| Identity and status | `status` (active / churned) |
| Segment and profile | `customer_segment`, `region`, `tenure_band`, `age_band`, `inactivity_band`, `complaint_band`, `product_band` |
| Model output | `churn_probability` (out-of-fold), `risk_tier`, `priority_tier` |
| Value | `balance_at_risk`, `annual_value_at_risk`, `net_retention_value`, `priority_rank` |
| Explanation and persona | `top_reasons`, `persona` |

Add `outputs/eda_tables.csv` for pre-computed churn rates with CIs.

## KPI cards (top row)

1. Overall churn: **15.8%**
2. Churned balance share: **6.6%**
3. Active balance at risk: sum of `balance_at_risk` where `status` = active
4. Customers in P1

## Filters (one row)

Segment, region, tenure band, RM (yes/no), priority tier, persona.

## Views

1. **"Who leaves":** churn rate by segment with CI bars, plus a churner count bar.
2. **"Complaints":** dose-response line (0 / 1 / 2 / 3+).
3. **"Balance at Risk":** churn rate by balance quintile next to share of churned balance (two panels, never a dual axis).
4. **"Silent Exits":** churn by inactivity band; KPI of active customers 46+ days inactive.
5. **Persona map:** scatter of `churn_probability` (y) vs log balance (x), coloured by persona (≤5 colours), with tier quadrant lines at the cut-offs in `outputs/tier_cutoffs.json`.
6. **WorthSaving matrix:** 2×2 of tiers (customer count, balance at risk).
7. **Top-50 priority list:** table sorted by `priority_rank` with reasons and recommended action.
8. **Gain curve:** churned balance captured vs customers contacted, for three ranking rules (from `prioritisation_results.json`).

## Wireframe

```
[ KPI | KPI | KPI | KPI ]
[ Filters ............................................ ]
[ Who leaves        ] [ Balance at Risk (2 panels)     ]
[ Complaints        ] [ Silent Exits                   ]
[ WorthSaving 2x2   ] [ Persona map                    ]
[ Top-50 priority list (full width)                    ]
```
