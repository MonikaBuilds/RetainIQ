"""RetainIQ - Stage 4: PRIORITISE ("WorthSaving").

Ranks ONLY currently active customers (churned = 0) using out-of-fold
(cross-fitted) calibrated probabilities. Churned customers are used for the
historical back-test of ranking rules, never as targets.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (CONTACT_COST, DATA, DEFAULT_CONTACT_COST, FEE_PER_PRODUCT, FIG, ID, NIM, OUT,
                    SAVE_RATE, TARGET, VALUE_HORIZON_YEARS, WINSOR_PCT)

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b1", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ececea", "axes.titleweight": "bold", "axes.titlesize": 10.5})
TIER_ACTION = {
    "P1": "Save now: RM call within 5 days + bespoke offer",
    "P2": "Nurture & deepen: cross-sell / loyalty, protect relationship",
    "P3": "Low-cost automation: digital nudge, no discount",
    "P4": "Monitor, do not spend",
}
TIER_COST = {"P1": CONTACT_COST["rm_call_offer"], "P2": CONTACT_COST["nurture"],
             "P3": CONTACT_COST["digital_nudge"], "P4": CONTACT_COST["monitor"]}


def value(df, nim=NIM, fee=FEE_PER_PRODUCT, bal_col="avg_monthly_balance"):
    return df[bal_col] * nim + df["num_products"] * fee


def play(r):
    """Driver-specific play from the evidence-based playbook (stage 5)."""
    plays = []
    if r["complaint_count_last_year"] >= 1:
        plays.append("complaint recovery (48h SLA + follow-up)")
    if r["last_transaction_days_ago"] > 45:
        plays.append("re-engagement journey")
    if r["num_products"] == 1:
        plays.append("second-product onboarding bundle")
    if r["rm_eligible_unassigned"] == 1:
        plays.append("assign RM / digital concierge")
    if r["tenure_years"] < 1:
        plays.append("90-day onboarding programme")
    if r["low_engagement"] == 1 and r["avg_monthly_balance"] >= 28796:
        plays.append("assisted digital onboarding")
    return "; ".join(plays[:2]) if plays else "relationship check-in"


def apply_tiers(df, cut):
    hr = df["churn_probability"] >= cut["high_risk_prob"]
    hv = df["avg_monthly_balance"] >= cut["high_value_balance"]
    df["priority_tier"] = np.select([hr & hv, ~hr & hv, hr & ~hv], ["P1", "P2", "P3"], "P4")
    df["risk_tier"] = np.select([df["churn_probability"] >= cut["high_risk_prob"],
                                 df["churn_probability"] >= cut["medium_risk_prob"]], ["High", "Medium"], "Low")
    return df


def score(df, nim=NIM, fee=FEE_PER_PRODUCT, save_rate=SAVE_RATE, bal_col="avg_monthly_balance", cost_override=None,
          horizon=VALUE_HORIZON_YEARS):
    d = df.copy()
    d["balance_at_risk"] = d["churn_probability"] * d[bal_col]
    d["annual_value"] = value(d, nim, fee, bal_col)
    d["annual_value_at_risk"] = d["churn_probability"] * d["annual_value"]
    cost = d["priority_tier"].map(TIER_COST) if cost_override is None else cost_override
    d["intervention_cost"] = cost
    # [ASSUMPTION] saving a customer preserves `horizon` years of margin
    d["net_retention_value"] = d["annual_value_at_risk"] * horizon * save_rate - cost
    d["priority_score"] = d["net_retention_value"]
    d = d.sort_values(["priority_score", "churn_probability"], ascending=False).reset_index(drop=True)
    d["priority_rank"] = np.arange(1, len(d) + 1)
    return d


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    df = pd.read_csv(DATA / "cleaned_bank_data.csv")
    oof = pd.read_csv(OUT / "oof_predictions.csv")
    per = pd.read_csv(OUT / "cluster_assignments.csv")[[ID, "persona"]]
    df = df.merge(oof, on=ID).merge(per, on=ID).rename(columns={"churn_probability_oof": "churn_probability"})
    R = {}

    # ---------------- historical back-test of ranking rules (all 10k, OOF probs, actual outcomes)
    df["value"] = value(df)
    rules = {"Probability only": df["churn_probability"],
             "Balance only": df["avg_monthly_balance"],
             "Probability x balance (RetainIQ)": df["churn_probability"] * df["avg_monthly_balance"]}
    churned_bal = (df[TARGET] * df["avg_monthly_balance"]).sum()
    bt = {}
    Ns = [100, 250, 500, 1000, 2000]
    for nm, s in rules.items():
        o = np.argsort(-s.values, kind="stable")
        y, b = df[TARGET].values[o], df["avg_monthly_balance"].values[o]
        bt[nm] = dict(curve_bal=np.cumsum(y * b) / churned_bal, curve_ch=np.cumsum(y) / y.sum(),
                      at_N={N: dict(churners=int(y[:N].sum()), churned_balance=float((y[:N] * b[:N]).sum()),
                                    share_churned_balance=float((y[:N] * b[:N]).sum() / churned_bal)) for N in Ns})
    R["backtest_all_10k"] = {nm: v["at_N"] for nm, v in bt.items()}
    R["backtest_note"] = ("Out-of-fold probabilities vs actual outcomes on all 10,000 customers: "
                          "which rule would have caught the most leaving money for a fixed contact budget?")

    # ---------------- rank ACTIVE customers only
    act = df[df[TARGET] == 0].copy()
    cut = dict(high_risk_prob=float(act["churn_probability"].quantile(0.75)),
               medium_risk_prob=float(act["churn_probability"].quantile(0.50)),
               high_value_balance=float(act["avg_monthly_balance"].quantile(0.75)),
               rule="top quartile of OOF probability / top quartile of balance among ACTIVE customers")
    json.dump(cut, open(OUT / "tier_cutoffs.json", "w"), indent=2)
    act = apply_tiers(act, cut)
    ranked = score(act)
    ranked["recommended_action"] = ranked["priority_tier"].map(TIER_ACTION) + " | " + ranked.apply(play, axis=1)
    cols = [ID, "priority_rank", "priority_tier", "risk_tier", "churn_probability", "avg_monthly_balance",
            "balance_at_risk", "annual_value", "annual_value_at_risk", "intervention_cost", "net_retention_value",
            "priority_score", "top_reasons", "persona", "customer_segment", "num_products",
            "complaint_count_last_year", "last_transaction_days_ago", "recommended_action"]
    ranked[cols].to_csv(OUT / "ranked_customers.csv", index=False)
    R["active_customers"] = len(ranked)
    R["tier_cutoffs"] = cut

    # 2x2 cell summary
    cell = ranked.groupby("priority_tier").agg(customers=(ID, "size"), mean_prob=("churn_probability", "mean"),
                                               expected_churners=("churn_probability", "sum"),
                                               total_balance=("avg_monthly_balance", "sum"),
                                               balance_at_risk=("balance_at_risk", "sum"),
                                               annual_value_at_risk=("annual_value_at_risk", "sum"),
                                               net_value=("net_retention_value", "sum"),
                                               positive_net=("net_retention_value", lambda x: int((x > 0).sum())))
    cell["share_balance_at_risk"] = cell["balance_at_risk"] / cell["balance_at_risk"].sum()
    cell.to_csv(OUT / "tier_summary.csv")
    R["tier_summary"] = cell.round(2).to_dict(orient="index")

    # ---------------- budget view on active customers (expected values)
    budget = {}
    for nm, key in (("Probability only", ranked["churn_probability"]),
                    ("Balance only", ranked["avg_monthly_balance"]),
                    ("Probability x balance (RetainIQ)", ranked["balance_at_risk"])):
        o = ranked.loc[np.argsort(-key.values, kind="stable")]
        budget[nm] = {}
        for N in (100, 200, 250, 500, 1000):
            t = o.head(N)
            budget[nm][N] = dict(expected_churners=float(t.churn_probability.sum()),
                                 expected_churners_saved=float(t.churn_probability.sum() * SAVE_RATE),
                                 balance_at_risk=float(t.balance_at_risk.sum()),
                                 expected_balance_protected=float(t.balance_at_risk.sum() * SAVE_RATE),
                                 expected_annual_value_protected=float(t.annual_value_at_risk.sum() * SAVE_RATE),
                                 expected_horizon_value_protected=float(t.annual_value_at_risk.sum() * SAVE_RATE * VALUE_HORIZON_YEARS),
                                 cost_at_default_contact=float(N * DEFAULT_CONTACT_COST),
                                 segment_mix=t.customer_segment.value_counts().to_dict())
    R["budget_active_expected"] = budget
    R["budget_assumptions"] = dict(save_rate=SAVE_RATE, nim=NIM, value_horizon_years=VALUE_HORIZON_YEARS, fee_per_product=FEE_PER_PRODUCT,
                                   default_contact_cost=DEFAULT_CONTACT_COST, tier_costs=TIER_COST)

    # top 200 for Q20
    top200 = ranked.head(200)
    R["top200"] = dict(tiers=top200.priority_tier.value_counts().to_dict(),
                       segments=top200.customer_segment.value_counts().to_dict(),
                       mean_prob=float(top200.churn_probability.mean()),
                       balance_at_risk=float(top200.balance_at_risk.sum()),
                       share_of_all_active_balance_at_risk=float(top200.balance_at_risk.sum() / ranked.balance_at_risk.sum()),
                       net_value=float(top200.net_retention_value.sum()),
                       top_actions=top200.recommended_action.str.split(" \\| ").str[1].value_counts().head(5).to_dict())

    # ---------------- heavy tail: concentration and winsorisation
    capv = act["avg_monthly_balance"].quantile(WINSOR_PCT)
    act_w = act.copy(); act_w["bal_w"] = act_w["avg_monthly_balance"].clip(upper=capv)
    rw = score(act_w, bal_col="bal_w")
    top100 = ranked.head(100)
    R["heavy_tail"] = dict(winsor_cap_99pct=float(capv),
                           largest_single_share_of_top100_bal_at_risk=float(top100.balance_at_risk.max() / top100.balance_at_risk.sum()),
                           jaccard_top250_raw_vs_winsor=jaccard(ranked.head(250)[ID], rw.head(250)[ID]),
                           jaccard_top1000_raw_vs_winsor=jaccard(ranked.head(1000)[ID], rw.head(1000)[ID]),
                           total_bal_at_risk_raw=float(ranked.balance_at_risk.sum()),
                           total_bal_at_risk_winsor=float(rw.balance_at_risk.sum()))

    # ---------------- sensitivity: how much does the top-250 list move?
    base_top = set(ranked.head(250)[ID])
    sens = []
    for lab, kw in (("value horizon 1 year", dict(horizon=1)), ("value horizon 5 years", dict(horizon=5)),
                    ("NIM halved (1.25%)", dict(nim=NIM / 2)), ("NIM +50% (3.75%)", dict(nim=NIM * 1.5)),
                    ("fee per product 0", dict(fee=0)), ("fee per product 100", dict(fee=100)),
                    ("save rate 10%", dict(save_rate=0.10)), ("save rate 30%", dict(save_rate=0.30)),
                    ("flat contact cost 10", dict(cost_override=10.0)), ("flat contact cost 50", dict(cost_override=50.0))):
        s = score(act, **kw)
        sens.append(dict(scenario=lab, jaccard_top250=jaccard(base_top, s.head(250)[ID]),
                         share_of_top250_changed=1 - len(base_top & set(s.head(250)[ID])) / 250,
                         n_positive_net_value=int((s.net_retention_value > 0).sum()),
                         total_net_value_top250=float(s.head(250).net_retention_value.sum())))
    sens.append(dict(scenario="balance winsorised at p99", jaccard_top250=jaccard(base_top, rw.head(250)[ID]),
                     share_of_top250_changed=1 - len(base_top & set(rw.head(250)[ID])) / 250,
                     n_positive_net_value=int((rw.net_retention_value > 0).sum()),
                     total_net_value_top250=float(rw.head(250).net_retention_value.sum())))
    # threshold sensitivity: risk cut-off at top 20% / 30% instead of 25%
    for q in (0.70, 0.80):
        c2 = dict(cut); c2["high_risk_prob"] = float(act["churn_probability"].quantile(q))
        t2 = apply_tiers(act.copy(), c2)
        p1a, p1b = set(ranked[ranked.priority_tier == "P1"][ID]), set(t2[t2.priority_tier == "P1"][ID])
        sens.append(dict(scenario=f"high-risk cut at top {1 - q:.0%}", jaccard_top250=np.nan,
                         p1_jaccard_vs_base=jaccard(p1a, p1b), p1_size=len(p1b)))
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "prioritisation_sensitivity.csv", index=False)
    R["sensitivity"] = sens.round(3).to_dict(orient="records")
    R["n_positive_net_value_base"] = int((ranked.net_retention_value > 0).sum())

    # ---------------- ethics: does tiering exclude any group?
    eth = {}
    for c in ("gender", "region", "age_band"):
        t = pd.crosstab(ranked[c], ranked.priority_tier, normalize="index")
        eth[c] = t.round(3).to_dict(orient="index")
    R["tier_mix_by_group"] = eth

    # ---------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    cols_ = {"Probability only": BLUE, "Balance only": GREY, "Probability x balance (RetainIQ)": ORANGE}
    xmax = 2000
    for nm, v in bt.items():
        axes[0].plot(np.arange(1, xmax + 1), v["curve_bal"][:xmax], color=cols_[nm], lw=2, label=nm)
        axes[1].plot(np.arange(1, xmax + 1), v["curve_ch"][:xmax], color=cols_[nm], lw=2, label=nm)
    for ax, t in ((axes[0], "Churned BALANCE caught"), (axes[1], "Churned CUSTOMERS caught")):
        ax.set_xlabel("Customers contacted (of 10,000, back-test)"); ax.set_title(t, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
        for N in (250, 1000):
            ax.axvline(N, color=GREY, lw=0.6, ls=":")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    a, b = bt["Probability x balance (RetainIQ)"]["at_N"][500], bt["Probability only"]["at_N"][500]
    fig.suptitle(f"Balance at Risk: for 500 contacts, RetainIQ's rule catches {a['share_churned_balance']:.0%} of the money that left "
                 f"vs {b['share_churned_balance']:.0%} for risk alone,\nwhile catching fewer customers "
                 f"({a['churners']} vs {b['churners']}) - it trades headcount for value",
                 x=0.02, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "17_budget_gain_curves.png", facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    pos = {"P1": (0, 1), "P2": (0, 0), "P3": (1, 1), "P4": (1, 0)}
    names = {"P1": "P1  Save now", "P2": "P2  Nurture & deepen", "P3": "P3  Low-cost automation", "P4": "P4  Monitor, don't spend"}
    colr = {"P1": ORANGE, "P2": BLUE, "P3": AQUA, "P4": GREY}
    for t, (x, yy) in pos.items():
        r = cell.loc[t]
        ax.add_patch(plt.Rectangle((x, yy), 0.97, 0.97, color=colr[t], alpha=0.18, lw=0))
        ax.text(x + 0.05, yy + 0.84, names[t], fontweight="bold", fontsize=10)
        ax.text(x + 0.05, yy + 0.12,
                f"{int(r.customers):,} customers\nmean P(churn) {r.mean_prob:.0%}\n"
                f"balance at risk {r.balance_at_risk / 1e6:.2f}m ({r.share_balance_at_risk:.0%})\n" +
                (f"net-positive to treat: {int(r.positive_net):,}" if t != "P4" else "no spend planned"),
                fontsize=8.5, va="bottom")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.grid(False); ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["High value (top 25% balance)", "Lower value"]); ax.set_yticklabels(["Lower risk", "High risk\n(top 25%)"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"WorthSaving matrix: {cell.loc['P1', 'customers']:,} P1 customers hold\n"
                 f"{cell.loc['P1', 'share_balance_at_risk']:.0%} of all balance at risk", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "18_worthsaving_matrix.png", facecolor="white"); plt.close(fig)

    # ---------------- dashboard extract (one row per customer, all 10k)
    all_t = apply_tiers(df.copy(), cut)
    all_s = score(all_t)
    all_s["status"] = np.where(all_s[TARGET] == 1, "churned", "active")
    all_s.loc[all_s.status == "churned", ["priority_tier", "priority_rank"]] = ["Churned (history)", np.nan]
    keep = [c for c in all_s.columns if c not in ("value",)]
    all_s[keep].to_csv(DATA / "dashboard_data.csv", index=False)

    checks = {"ranked list contains only active customers": set(ranked[ID]).isdisjoint(set(df.loc[df[TARGET] == 1, ID])),
              "ranked list size == active customers": len(ranked) == int((df[TARGET] == 0).sum()),
              "probabilities are out-of-fold": "churn_probability_oof" in oof.columns,
              "all tiers populated": set(ranked.priority_tier) == {"P1", "P2", "P3", "P4"}}
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    R["checks"] = checks
    json.dump(R, open(OUT / "prioritisation_results.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(cell)
    print(pd.DataFrame({k: {N: round(v[N]["share_churned_balance"], 3) for N in Ns} for k, v in R["backtest_all_10k"].items()}))
    print(sens.to_string())
    assert all(checks.values())


if __name__ == "__main__":
    main()

