"""RetainIQ - Stage 5: ACT. Pilot sizing (power) and reproducible randomised assignment.

The model predicts WHO is likely to leave, not who is PERSUADABLE. Only a
randomised hold-out control can measure true uplift.
"""
import json

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from config import ID, OUT, SEED

ALPHA, POWER = 0.05, 0.80


def n_per_arm(p0, mde):
    es = proportion_effectsize(p0, p0 - mde)
    return int(np.ceil(NormalIndPower().solve_power(es, alpha=ALPHA, power=POWER, ratio=1.0)))


def mde_for_n(p0, n):
    lo, hi = 1e-4, p0 - 1e-4
    for _ in range(60):
        mid = (lo + hi) / 2
        if n_per_arm(p0, mid) > n:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    r = pd.read_csv(OUT / "ranked_customers.csv")
    P = {"alpha": ALPHA, "power": POWER,
         "reference_example": dict(baseline=0.20, mde=0.04, n_per_arm=n_per_arm(0.20, 0.04))}
    groups = {"P1+P2 (high value)": r[r.priority_tier.isin(["P1", "P2"])],
              "P1+P3 (high risk)": r[r.priority_tier.isin(["P1", "P3"])],
              "Top 40% risk (all values)": r[r.churn_probability >= r.churn_probability.quantile(0.60)]}
    rows = []
    for g, d in groups.items():
        p0 = float(d.churn_probability.mean())
        rows.append(dict(group=g, customers=len(d), baseline_pred_churn=p0,
                         n_per_arm_for_4pp=n_per_arm(p0, 0.04) if p0 > 0.05 else None,
                         n_per_arm_for_20pct_relative=n_per_arm(p0, 0.2 * p0),
                         mde_with_available_n=mde_for_n(p0, len(d) // 2)))
    sizing = pd.DataFrame(rows)
    sizing.to_csv(OUT / "pilot_sizing.csv", index=False)
    P["sizing"] = sizing.round(4).to_dict(orient="records")

    # Recommended pilot population: high-risk cells (P1+P3). Rationale: P2's baseline
    # churn (~4%) is too low to detect any realistic retention effect in 90-180 days.
    pilot = groups["P1+P3 (high risk)"].copy()
    pilot["risk_decile"] = pd.qcut(pilot.churn_probability.rank(method="first"), 10, labels=False)
    pilot["stratum"] = pilot.priority_tier + "|" + pilot.customer_segment + "|d" + pilot.risk_decile.astype(str)
    rng = np.random.default_rng(SEED)
    arm = np.empty(len(pilot), dtype=object)
    for _, idx in pilot.groupby("stratum").indices.items():
        idx = rng.permutation(idx)
        half = len(idx) // 2 + (rng.random() < 0.5 if len(idx) % 2 else 0)
        arm[idx[:half]] = "treatment"; arm[idx[half:]] = "control"
    pilot["arm"] = arm
    pilot[[ID, "priority_tier", "customer_segment", "risk_decile", "stratum", "arm",
           "churn_probability", "avg_monthly_balance"]].to_csv(OUT / "pilot_assignment.csv", index=False)
    bal = pilot.groupby("arm").agg(n=(ID, "size"), mean_pred=("churn_probability", "mean"),
                                   mean_balance=("avg_monthly_balance", "mean"),
                                   p1_share=("priority_tier", lambda x: (x == "P1").mean()),
                                   mass_market_share=("customer_segment", lambda x: (x == "Mass Market").mean()))
    P["recommended_population"] = "P1 + P3 (high-risk quartile), stratified by tier x segment x risk decile"
    P["arms"] = bal.round(4).to_dict(orient="index")
    p0 = float(pilot.churn_probability.mean())
    P["achievable_mde_pp"] = mde_for_n(p0, int(bal.n.min()))
    P["note"] = ("Predicted churn is a historical-snapshot rate, not a 90/180-day rate; re-estimate the "
                 "baseline from the control arm's first 30 days and extend the window if needed.")
    json.dump(P, open(OUT / "pilot_plan.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(sizing.to_string()); print(bal); print("achievable MDE:", round(P["achievable_mde_pp"], 4))
    print(f"[{'PASS' if abs(bal.n.iloc[0] - bal.n.iloc[1]) <= bal.n.sum() * 0.02 else 'FAIL'}] arms balanced in size")


if __name__ == "__main__":
    main()
