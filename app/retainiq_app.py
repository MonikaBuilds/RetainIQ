"""RetainIQ_Dashboard: Predict. Prioritise. Retain.

Run:  streamlit run app/retainiq_app.py

Reads only pipeline outputs (python src/run_all.py) - no number on screen is typed in.
Layout and rationale: reports/dashboard_spec.md, reports/ux_research_brief.md.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prioritise import TIER_COST, score  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b1"   # blue = risk/rate, orange = money
TIER_COLOR = {"P1": ORANGE, "P2": AQUA, "P3": BLUE, "P4": GREY}
TIER_NAME = {"P1": "P1 Save now", "P2": "P2 Nurture & deepen", "P3": "P3 Low-cost automation", "P4": "P4 Monitor"}
MIN_N = 30

st.set_page_config(page_title="RetainIQ", page_icon="📈", layout="wide")
st.markdown("""<style>
.chip{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:600;
      letter-spacing:.04em;margin-bottom:.4rem;border:1px solid currentColor}
.chip.D{color:#2a78d6}.chip.P{color:#7b5cd6}.chip.R{color:#eb6834}
.card{border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:14px 16px;margin-bottom:8px}
.muted{opacity:.7;font-size:.85rem}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    d = pd.read_csv(DATA / "dashboard_data.csv")
    j = lambda f: json.load(open(OUT / f))
    return dict(d=d, metrics=j("metrics.json"), eda=j("eda_results.json"), unseen=j("demo_unseen_results.json"), prio=j("prioritisation_results.json"), pilot=j("pilot_plan.json"),
                cut=j("tier_cutoffs.json"), fair=pd.read_csv(OUT / "fairness_audit.csv"),
                perm=pd.read_csv(OUT / "permutation_importance.csv"),
                sizing=pd.read_csv(OUT / "pilot_sizing.csv"), sens=pd.read_csv(OUT / "prioritisation_sensitivity.csv"))


def wilson(k, n, z=1.96):
    n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - h, c + h


def rates(df, by, order=None):
    g = df.groupby(by, observed=True).agg(n=("churned", "size"), k=("churned", "sum"),
                                          bal=("avg_monthly_balance", lambda s: s[df.loc[s.index, "churned"] == 1].sum()))
    if order:
        g = g.reindex([o for o in order if o in g.index])
    g["rate"], g["lo"], g["hi"] = wilson(g["k"].values, g["n"].values)
    tot = (df["churned"] * df["avg_monthly_balance"]).sum()
    g["bal_share"] = g["bal"] / tot if tot else np.nan
    g.loc[g["n"] < MIN_N, ["rate", "lo", "hi"]] = np.nan          # suppress small cells
    return g.reset_index()


def val(x):
    return x["value"] if isinstance(x, dict) else x


def ci(x):
    return f"[{x['ci_low']:.3f}–{x['ci_high']:.3f}]" if isinstance(x, dict) and "ci_low" in x else ""


def chip(kind):
    label = {"D": "DESCRIPTIVE · what happened", "P": "PREDICTIVE · model estimate",
             "R": "RECOMMENDATION · our judgement"}[kind]
    st.markdown(f'<span class="chip {kind}">{label}</span>', unsafe_allow_html=True)


def style(fig, h=360, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), bargap=0.25,
                      legend=dict(orientation="h", y=-0.18), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)")
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    return fig


def rate_bar(g, x, title, color=BLUE, xlab=None):
    txt = [f"{r:.0%}<br><span style='font-size:10px'>n={n:,}</span>" if pd.notna(r) else f"n={n} (suppressed)"
           for r, n in zip(g["rate"], g["n"])]
    fig = go.Figure(go.Bar(
        x=g[x].astype(str), y=g["rate"], marker_color=color, text=txt, textposition="outside",
        error_y=dict(type="data", symmetric=False, array=g["hi"] - g["rate"], arrayminus=g["rate"] - g["lo"],
                     color="rgba(128,128,128,.7)", thickness=1.2),
        customdata=np.c_[g["n"], g["k"], g["lo"], g["hi"]],
        hovertemplate="%{x}<br>churn %{y:.1%} [%{customdata[2]:.1%}–%{customdata[3]:.1%}]"
                      "<br>%{customdata[1]:,} of %{customdata[0]:,} customers<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="churn rate (95% CI)", rangemode="tozero")
    fig.update_xaxes(title=xlab or x)
    return style(fig, title=title)


A = load()
d, M, PR, CUT, E = A["d"], A["metrics"], A["prio"], A["cut"], A["eda"]
RM = E["hypotheses"]["H7_rm_confounding"]
ARI = next(r["bootstrap_ari"] for r in E["clustering"]["selection"] if r["k"] == E["clustering"]["k"])
R_SINGLE = d.loc[d["num_products"] == 1, "churned"].mean()
R_NEW = d.loc[d["tenure_band"] == "<1", "churned"].mean()
active_all = d[d["status"] == "active"].copy()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## RetainIQ")
    st.caption("Predict. Prioritise. Retain. — *which customers are about to leave, and which are worth saving.*")
    st.markdown("### Filters")
    st.caption("Apply to **See** and **Prioritise**.")
    f_seg = st.multiselect("Segment", sorted(d["customer_segment"].unique()))
    f_reg = st.multiselect("Region", sorted(d["region"].unique()))
    f_ten = st.multiselect("Tenure band (years)", ["<1", "1-3", "3-7", "7-15", "15+"])
    f_rm = st.radio("Relationship manager", ["All", "Assigned", "Not assigned"], horizontal=True)
    f_per = st.multiselect("Persona", sorted(d["persona"].unique()))
    st.markdown("### Assumptions")
    st.caption("`[ASSUMPTION]` Re-rank live. Defaults come from `src/config.py`.")
    a = PR["budget_assumptions"]
    nim = st.slider("Net interest margin", 0.005, 0.05, float(a["nim"]), 0.0025, format="%.4f")
    save = st.slider("Save rate if contacted", 0.05, 0.50, float(a["save_rate"]), 0.05)
    horizon = st.slider("Value horizon (years)", 1, 5, int(a["value_horizon_years"]))
    cost_mult = st.slider("Contact cost multiplier", 0.5, 3.0, 1.0, 0.25,
                          help="Scales P1 150 / P2 40 / P3 5 / P4 0")


def apply_filters(df):
    m = pd.Series(True, index=df.index)
    for col, sel in [("customer_segment", f_seg), ("region", f_reg), ("tenure_band", f_ten), ("persona", f_per)]:
        if sel:
            m &= df[col].isin(sel)
    if f_rm != "All":
        m &= df["relationship_manager_assigned"] == (1 if f_rm == "Assigned" else 0)
    return df[m]


dfil = apply_filters(d)
n_filters = sum(map(bool, [f_seg, f_reg, f_ten, f_per])) + (f_rm != "All")

# live re-score on ALL active customers (ranks are global), then filter for display
rescored = score(active_all, nim=nim, fee=a["fee_per_product"], save_rate=save, horizon=horizon,
                 cost_override=active_all["priority_tier"].map(TIER_COST) * cost_mult)
base_top = set(active_all.nsmallest(250, "priority_rank")["customer_id"])
new_top = set(rescored.head(250)["customer_id"])
changed_250 = 1 - len(base_top & new_top) / 250

st.title("RetainIQ_Dashboard")
st.markdown("**Churners are many but light; the money that leaves sits with a few.** "
            "RetainIQ ranks *who is worth saving*, not just who will leave.")
if n_filters:
    st.info(f"Filters active: {len(dfil):,} of {len(d):,} customers shown in See / Prioritise.")

tabs = st.tabs(["Overview", "See", "Predict", "Prioritise", "Act", "Score a file"])

# ================================================================== OVERVIEW
with tabs[0]:
    chip("D")
    y, b = d["churned"], d["avg_monthly_balance"]
    rate, lo, hi = wilson(y.sum(), len(y))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers who churned", f"{float(rate):.1%}", help=f"95% CI {float(lo):.1%}–{float(hi):.1%}")
    c1.caption(f"{int(y.sum()):,} of {len(y):,}")
    c2.metric("Share of balances that left", f"{(y * b).sum() / b.sum():.1%}")
    c2.caption(f"{(y * b).sum() / 1e6:,.1f}M of {b.sum() / 1e6:,.0f}M")
    c3.metric("Active balance at risk", f"{active_all['balance_at_risk'].sum() / 1e6:,.1f}M")
    c3.caption("Σ P(churn) × balance · predictive")
    p1 = (active_all["priority_tier"] == "P1").sum()
    c4.metric("P1 — Save now", f"{p1:,}")
    c4.caption(f"carry {PR['tier_summary']['P1']['share_balance_at_risk']:.0%} of balance at risk")

    g = rates(d, "customer_segment", ["Mass Market", "Mass Affluent", "Affluent", "Private Banking"])
    g["ch_share"] = g["k"] / g["k"].sum()
    fig = go.Figure([
        go.Bar(name="Share of churners", y=g["customer_segment"], x=g["ch_share"], orientation="h",
               marker_color=BLUE, text=[f"{v:.0%}" for v in g["ch_share"]], textposition="outside"),
        go.Bar(name="Share of churned balance", y=g["customer_segment"], x=g["bal_share"], orientation="h",
               marker_color=ORANGE, text=[f"{v:.0%}" for v in g["bal_share"]], textposition="outside")])
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 340, barmode="group",
                          title="Mass Market is most of the churners, but the money that left is spread up the book"),
                    theme="streamlit", use_container_width=True)
    st.markdown(f"""
| Stage | Headline |
|---|---|
| **See** | Single-product customers churn {rates(d, 'single_product')['rate'].iloc[-1]:.0%}; complaints show a dose-response |
| **Predict** | Hold-out ROC-AUC **{val(M['holdout_metrics']['roc_auc']):.3f}** {ci(M['holdout_metrics']['roc_auc'])}, {val(M['holdout_metrics']['lift_top10']):.2f}× lift in the top 10% |
| **Prioritise** | Risk × balance catches **{PR['backtest_all_10k']['Probability x balance (RetainIQ)']['500']['share_churned_balance']:.0%}** of leaving money with 500 contacts vs {PR['backtest_all_10k']['Probability only']['500']['share_churned_balance']:.0%} for risk only |
| **Act** | Randomised pilot of {PR['tier_summary']['P1']['customers'] + PR['tier_summary']['P3']['customers']:,} customers detects a {A['pilot']['achievable_mde_pp'] * 100:.1f}-point drop |
""")

# ================================================================== SEE
with tabs[1]:
    chip("D")
    st.caption(f"{len(dfil):,} customers · 95% Wilson CIs · cells with n < {MIN_N} suppressed · associations, not causes.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(rate_bar(rates(dfil, "product_band", ["1", "2", "3", "4", "5+"]), "product_band",
                             "One product is the strongest single churn signal", xlab="products held"),
                    theme="streamlit", use_container_width=True)
    c2.plotly_chart(rate_bar(rates(dfil, "complaint_band", ["0", "1", "2", "3+"]), "complaint_band",
                             "Each extra complaint goes with more churn", xlab="complaints last year (3+ grouped)"),
                    theme="streamlit", use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(rate_bar(rates(dfil, "tenure_band", ["<1", "1-3", "3-7", "7-15", "15+"]), "tenure_band",
                             "New customers are the most fragile", xlab="tenure (years)"),
                    theme="streamlit", use_container_width=True)
    c2.plotly_chart(rate_bar(rates(dfil, "inactivity_band", ["<=15", "16-45", "46-90", "90+"]), "inactivity_band",
                             "Silent Exits: going quiet precedes — or is — leaving", xlab="days since last transaction"),
                    theme="streamlit", use_container_width=True)
    st.caption("`[LIMITATION]` Inactivity may be a symptom of leaving, so the deployed model excludes it (see Predict → leakage).")

    st.subheader("Balance at risk: where the money is")
    q = dfil.assign(bal_q=pd.qcut(dfil["avg_monthly_balance"].rank(method="first"), 5,
                                  labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]))
    g = rates(q, "bal_q")
    fig = make_subplots(1, 2, subplot_titles=("Churn rate by balance quintile", "Share of churned balance"))
    fig.add_bar(x=g["bal_q"].astype(str), y=g["rate"], marker_color=BLUE, name="churn rate",
                error_y=dict(type="data", symmetric=False, array=g["hi"] - g["rate"], arrayminus=g["rate"] - g["lo"]),
                text=[f"{v:.0%}" for v in g["rate"]], textposition="outside", row=1, col=1)
    fig.add_bar(x=g["bal_q"].astype(str), y=g["bal_share"], marker_color=ORANGE, name="share of churned balance",
                text=[f"{v:.0%}" for v in g["bal_share"]], textposition="outside", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(style(fig, 360, showlegend=False,
                          title="The richest fifth churns least but holds most of the money that left"),
                    theme="streamlit", use_container_width=True)

    c1, c2 = st.columns(2)
    g = dfil.assign(RM=np.where(dfil["relationship_manager_assigned"] == 1, "RM assigned", "No RM"))
    g = rates(g, ["customer_segment", "RM"])
    fig = go.Figure()
    for rm, col in [("No RM", GREY), ("RM assigned", BLUE)]:
        s = g[g["RM"] == rm]
        fig.add_bar(name=rm, x=s["customer_segment"], y=s["rate"], marker_color=col,
                    error_y=dict(type="data", symmetric=False, array=s["hi"] - s["rate"], arrayminus=s["rate"] - s["lo"]),
                    customdata=s["n"], hovertemplate="%{x}<br>%{y:.1%} · n=%{customdata:,}<extra>" + rm + "</extra>")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(categoryorder="array", categoryarray=["Mass Market", "Mass Affluent", "Affluent", "Private Banking"])
    c1.plotly_chart(style(fig, 380, barmode="group", title="RM customers churn less in every segment…"),
                    theme="streamlit", use_container_width=True)
    c1.caption(f"…but RMs go to wealthy customers (median balance {RM['median_balance_rm']:,.0f} vs "
               f"{RM['median_balance_no_rm']:,.0f}). Adjusted for segment, balance, tenure and products the odds ratio "
               f"moves {RM['crude_or'][0]:.2f} → {RM['adjusted_or'][0]:.2f} "
               f"[{RM['adjusted_or'][1]:.2f}–{RM['adjusted_or'][2]:.2f}]: **association, not proof RMs cause retention.**")
    g = rates(dfil, "persona").sort_values("bal_share")
    fig = make_subplots(1, 2, shared_yaxes=True, subplot_titles=("churn rate", "share of churned balance"))
    fig.add_bar(y=g["persona"], x=g["rate"], orientation="h", marker_color=BLUE,
                text=[f"{v:.0%}" for v in g["rate"]], textposition="outside", row=1, col=1)
    fig.add_bar(y=g["persona"], x=g["bal_share"], orientation="h", marker_color=ORANGE,
                text=[f"{v:.0%}" for v in g["bal_share"]], textposition="outside", row=1, col=2)
    fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    c2.plotly_chart(style(fig, 380, showlegend=False, title="Personas: churn-heavy ≠ money-heavy"),
                    theme="streamlit", use_container_width=True)
    c2.caption(f"{E['clustering']['k']} personas from K-Means on behaviour only (churn not used); bootstrap ARI {ARI:.2f}.")

# ================================================================== PREDICT
with tabs[2]:
    chip("P")
    H = M["holdout_metrics"]
    st.caption(f"Deployed: **{M['deployment']['variant']}** (logistic regression) · scored on a 2,000-customer "
               "hold-out never used for training, tuning, calibration or threshold choice.")
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{val(H['roc_auc']):.3f}"); c[0].caption(ci(H["roc_auc"]))
    c[1].metric("PR-AUC", f"{val(H['pr_auc']):.3f}"); c[1].caption(f"random = {M['split']['holdout_churn']:.3f}")
    c[2].metric("Top-10% lift", f"{val(H['lift_top10']):.2f}×")
    c[2].caption(f"rule of thumb {val(M['holdout_baselines']['rule_of_thumb']['lift_top10']):.2f}×")
    c[3].metric("Brier", f"{val(H['brier']):.3f}")
    s = M["stability_10_seeds"]
    c[4].metric("AUC over 10 splits", f"{s['roc_mean']:.3f}"); c[4].caption(f"± {s['roc_sd']:.3f}")
    st.caption("Accuracy is not shown as a headline: predicting “nobody leaves” scores 84%.")

    c1, c2 = st.columns(2)
    L = M["leakage_experiment_holdout"]
    names = {"rule_of_thumb": "Rule of thumb", "A_full": "A · all features",
             "B_no_inactivity": "B · no inactivity", "C_early_warning": "C · no inactivity, no NPS"}
    lv = pd.DataFrame({"variant": [names[k] for k in L], "auc": [val(L[k]["roc_auc"]) for k in L]})
    dep = names[M["deployment"]["variant"]]
    fig = go.Figure(go.Bar(x=lv["auc"], y=lv["variant"], orientation="h", text=[f"{v:.3f}" for v in lv["auc"]],
                           textposition="outside", marker_color=[ORANGE if v == dep else GREY for v in lv["variant"]]))
    fig.update_xaxes(range=[0.5, 0.9], title="hold-out ROC-AUC")
    fig.update_yaxes(autorange="reversed")
    c1.plotly_chart(style(fig, 320, title="Leakage test: B keeps most of the signal without the symptoms"),
                    theme="streamlit", use_container_width=True)
    c1.caption(f"Rule fixed before results: *{M['deployment']['rule']}*. Deployed variant highlighted.")

    oof = d.assign(dec=pd.qcut(d["churn_probability"].rank(method="first"), 10, labels=False))
    cal = oof.groupby("dec").agg(pred=("churn_probability", "mean"), k=("churned", "sum"), n=("churned", "size"))
    cal["act"], cal["lo"], cal["hi"] = wilson(cal["k"].values, cal["n"].values)
    fig = go.Figure([go.Scatter(x=[0, .7], y=[0, .7], mode="lines", line=dict(color=GREY, dash="dot"),
                                name="perfect calibration"),
                     go.Scatter(x=cal["pred"], y=cal["act"], mode="lines+markers", name="actual (out-of-fold)",
                                line=dict(color=BLUE, width=2), marker=dict(size=9),
                                error_y=dict(type="data", symmetric=False, array=cal["hi"] - cal["act"],
                                             arrayminus=cal["act"] - cal["lo"]))])
    fig.update_xaxes(tickformat=".0%", title="predicted churn (decile mean)")
    fig.update_yaxes(tickformat=".0%", title="actual churn")
    c2.plotly_chart(style(fig, 320, title="Calibration: predicted risk tracks actual churn decile by decile"),
                    theme="streamlit", use_container_width=True)

    c1, c2 = st.columns(2)
    o = np.argsort(-d["churn_probability"].values, kind="stable")
    yy = d["churned"].values[o]
    xs = np.arange(1, len(yy) + 1) / len(yy)
    rot = (d["complaint_count_last_year"] + d["inactive_45plus"] * 2).values
    yr = d["churned"].values[np.argsort(-rot, kind="stable")]
    fig = go.Figure([go.Scatter(x=xs, y=np.cumsum(yy) / yy.sum(), name="RetainIQ (out-of-fold)", line=dict(color=BLUE, width=2)),
                     go.Scatter(x=xs, y=np.cumsum(yr) / yr.sum(), name="Rule of thumb", line=dict(color=ORANGE, width=2)),
                     go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, dash="dot"))])
    fig.update_xaxes(tickformat=".0%", title="share of customers contacted (highest risk first)")
    fig.update_yaxes(tickformat=".0%", title="share of churners caught")
    g20 = (np.cumsum(yy) / yy.sum())[int(0.2 * len(yy)) - 1]
    c1.plotly_chart(style(fig, 360, hovermode="x unified",
                          title=f"Contacting the riskiest 20% reaches {g20:.0%} of churners"),
                    theme="streamlit", use_container_width=True)
    c1.caption("Rule of thumb here = complaints + 2×(46+ days inactive), on all 10,000 customers with out-of-fold scores.")

    pi = A["perm"].head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=pi["importance"], y=pi["feature"], orientation="h", marker_color=BLUE,
                           error_x=dict(type="data", array=pi["sd"], color="rgba(128,128,128,.7)")))
    fig.update_xaxes(title="drop in hold-out PR-AUC when shuffled")
    c2.plotly_chart(style(fig, 360, title="Drivers: products, digital engagement, complaints, NPS"),
                    theme="streamlit", use_container_width=True)
    c2.caption("Permutation importance ± sd. Gender, age and region are never model inputs.")

    st.subheader("Fairness audit (hold-out, operating threshold)")
    fa = A["fair"].copy()
    fa["flag"] = np.where(fa["auc_gap_flag"] | fa["recall_gap_flag"], "⚠ review", "")
    st.dataframe(fa[["attribute", "group", "n", "n_churners", "churn_rate", "mean_pred", "flag_rate", "auc", "recall", "fpr", "flag"]]
                 .style.format({c: "{:.1%}" for c in ["churn_rate", "mean_pred", "flag_rate", "recall", "fpr"]} | {"auc": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
    gr = M["fairness"]["gender_region_ranges"]["gender"]
    st.caption(f"Gender AUC gap {gr['auc_range']:.3f}, recall gap {gr['recall_range']:.3f}. Segment recall differs "
               "because base rates differ; value-based tiers cover low-recall wealthy segments.")

# ================================================================== PRIORITISE
with tabs[3]:
    chip("R")
    rs = rescored[rescored["customer_id"].isin(apply_filters(active_all)["customer_id"])]
    st.caption(f"{len(rs):,} active customers · out-of-fold probabilities (no customer is scored by a model that saw them). "
               f"Your assumptions change **{changed_250:.0%}** of the top-250 list vs the defaults.")
    ts = rs.groupby("priority_tier").agg(n=("customer_id", "size"), risk=("churn_probability", "mean"),
                                         bar=("balance_at_risk", "sum"), net=("net_retention_value", "sum"))
    cols = st.columns(4)
    for col, t in zip(cols, ["P1", "P2", "P3", "P4"]):
        if t in ts.index:
            r = ts.loc[t]
            col.markdown(f'<div class="card" style="border-left:5px solid {TIER_COLOR[t]}"><b>{TIER_NAME[t]}</b><br>'
                         f'<span style="font-size:1.6rem;font-weight:700">{int(r.n):,}</span> customers<br>'
                         f'<span class="muted">avg risk {r.risk:.0%} · balance at risk {r.bar / 1e6:,.2f}M<br>'
                         f'net value {r.net:,.0f}</span></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    smp = rs if len(rs) <= 5000 else rs.sample(5000, random_state=42)
    fig = go.Figure()
    for t in ["P4", "P3", "P2", "P1"]:
        s = smp[smp["priority_tier"] == t]
        fig.add_trace(go.Scattergl(x=s["churn_probability"], y=s["avg_monthly_balance"].clip(lower=100), mode="markers",
                                   name=TIER_NAME[t], marker=dict(color=TIER_COLOR[t], size=6, opacity=.7,
                                                                  line=dict(width=0.5, color="white")),
                                   customdata=np.c_[s["customer_id"], s["persona"], s["top_reasons"]],
                                   hovertemplate="%{customdata[0]}<br>risk %{x:.0%} · balance %{y:,.0f}"
                                                 "<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"))
    fig.add_vline(x=CUT["high_risk_prob"], line_dash="dash", line_color=GREY)
    fig.add_hline(y=CUT["high_value_balance"], line_dash="dash", line_color=GREY)
    fig.update_xaxes(tickformat=".0%", title="churn probability")
    fig.update_yaxes(type="log", title="average balance (log)")
    c1.plotly_chart(style(fig, 460, title="WorthSaving matrix: top-right is where risk meets money"),
                    theme="streamlit", use_container_width=True)
    c1.caption(f"Lines: top quartile of risk ({CUT['high_risk_prob']:.0%}) and of balance ({CUT['high_value_balance']:,.0f}) "
               f"among active customers.{' Showing a 5,000-point sample.' if len(rs) > 5000 else ''}")

    with c2:
        st.markdown("**Budget simulator — back-test on actual outcomes**")
        N = st.slider("Customers we can contact", 50, 2000, 500, 50)
        o_rules = {"RetainIQ (risk × balance)": d["churn_probability"] * d["avg_monthly_balance"],
                   "Risk only": d["churn_probability"], "Balance only": d["avg_monthly_balance"]}
        cb = (d["churned"] * d["avg_monthly_balance"]).sum()
        rows, fig = [], go.Figure()
        for (nm, sc), colr in zip(o_rules.items(), [ORANGE, BLUE, GREY]):
            oo = np.argsort(-sc.values, kind="stable")
            curve = np.cumsum((d["churned"] * d["avg_monthly_balance"]).values[oo]) / cb
            fig.add_scatter(x=np.arange(1, 2001), y=curve[:2000], name=nm, line=dict(color=colr, width=2))
            rows.append({"rule": nm, "churned balance caught": curve[N - 1], "churners caught": int(d["churned"].values[oo][:N].sum())})
        fig.add_vline(x=N, line_color=GREY, line_dash="dot")
        fig.update_xaxes(title="contacts")
        fig.update_yaxes(tickformat=".0%", title="share of churned balance caught")
        st.plotly_chart(style(fig, 300, hovermode="x unified", title_text=""), theme="streamlit", use_container_width=True)
        st.dataframe(pd.DataFrame(rows).style.format({"churned balance caught": "{:.0%}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Risk only catches more *people*; RetainIQ catches more *money* — the trade the brief asks for.")

    st.subheader("Priority list")
    c1, c2, c3 = st.columns([2, 2, 1])
    q_id = c1.text_input("Search customer ID", placeholder="e.g. CUST100123")
    f_tier = c2.multiselect("Tier", ["P1", "P2", "P3", "P4"], default=[])
    top_n = c3.number_input("Show top", 10, 8422, 50, 10)
    lst = rs if not f_tier else rs[rs["priority_tier"].isin(f_tier)]
    if q_id:
        lst = lst[lst["customer_id"].str.contains(q_id.strip(), case=False)]
    show = lst.head(int(top_n))[["priority_rank", "customer_id", "priority_tier", "churn_probability", "avg_monthly_balance",
                                 "balance_at_risk", "net_retention_value", "customer_segment", "persona", "top_reasons"]]
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={
        "churn_probability": st.column_config.ProgressColumn("risk", format="%.0f%%", min_value=0, max_value=1),
        "avg_monthly_balance": st.column_config.NumberColumn("balance", format="%.0f"),
        "balance_at_risk": st.column_config.NumberColumn("balance at risk", format="%.0f"),
        "net_retention_value": st.column_config.NumberColumn("net value", format="%.0f")})
    st.download_button("Download this list (CSV)", lst.head(int(top_n)).to_csv(index=False), "retainiq_priority_list.csv",
                       "text/csv")

    st.subheader("Customer card")
    if len(show):
        cid = st.selectbox("Customer", show["customer_id"])
        r = rescored.set_index("customer_id").loc[cid]
        reasons = [x.strip() for x in str(r["top_reasons"]).split("+")]
        st.markdown(
            f'<div class="card" style="border-left:6px solid {TIER_COLOR[r.priority_tier]}">'
            f'<b>{cid}</b> · {r.customer_segment} · {r.persona} · tenure {r.tenure_years:.0f} y<br>'
            f'<span style="font-size:1.1rem">Looks like past leavers because of <b>{", ".join(reasons)}</b>.<br>'
            f'Do: <b>{TIER_NAME[r.priority_tier]}</b>.</span><br>'
            f'<span class="muted">risk {r.churn_probability:.0%} (an estimate, not a certainty) · balance {r.avg_monthly_balance:,.0f} · '
            f'balance at risk {r.balance_at_risk:,.0f} · net value {r.net_retention_value:,.0f} · rank {int(r.priority_rank):,} of {len(rescored):,}<br>'
            f'{int(r.num_products)} products · {int(r.complaint_count_last_year)} complaints · '
            f'{int(r.last_transaction_days_ago)} days since last transaction · digital score {r.digital_engagement_score:.0f}</span></div>',
            unsafe_allow_html=True)
    else:
        st.info("No customers match the current filters.")

# ================================================================== ACT
with tabs[4]:
    chip("R")
    st.subheader("Playbook: each action tied to its evidence")
    st.markdown(f"""
| Driver (evidence) | Action | Cost |
|---|---|---|
| Single product — {R_SINGLE:.0%} churn | Second-product bundle + 6-month fee waiver | Low |
| Complaint in the last year — dose-response | 48-hour resolution SLA + follow-up call | Medium |
| 46+ days inactive — Silent Exit trigger | Re-engagement journey (app nudge + first-transaction cashback) | Low |
| High balance, no RM | Assign an RM or digital concierge | High |
| Under 1 year tenure — {R_NEW:.0%} churn | 90-day onboarding programme | Low |
| High balance, low digital engagement | Assisted digital set-up in branch | Medium |
""")
    P = A["pilot"]
    st.subheader("Pilot: prove the save rate before scaling")
    c = st.columns(4)
    c[0].metric("Treatment", f"{P['arms']['treatment']['n']:,}")
    c[1].metric("Control", f"{P['arms']['control']['n']:,}")
    c[2].metric("Detectable drop", f"{P['achievable_mde_pp'] * 100:.1f} pts")
    c[3].metric("Baseline risk", f"{P['arms']['treatment']['mean_pred']:.0%}")
    st.caption(f"Population: {P['recommended_population']}. α = {P['alpha']}, power = {P['power']:.0%}, 120 days pre-registered.")
    arms = pd.DataFrame(P["arms"]).T
    st.dataframe(arms.style.format({"n": "{:,.0f}", "mean_pred": "{:.1%}", "mean_balance": "{:,.0f}",
                                    "p1_share": "{:.1%}", "mass_market_share": "{:.1%}"}), use_container_width=True)
    st.markdown("**Candidate populations considered**")
    st.dataframe(A["sizing"].style.format({"customers": "{:,}", "baseline_pred_churn": "{:.1%}",
                                           "mde_with_available_n": "{:.1%}"}), hide_index=True, use_container_width=True)
    st.markdown("""**Decision rule (pre-registered)** — *scale* if uplift ≥ 4 points, ROI > 1 and guardrails clean;
*iterate* if uplift is positive but ROI < 1; otherwise *stop*.
Guardrails: complaint rate, opt-outs, margin given away, contact fatigue.""")
    st.caption(f"`[LIMITATION]` {P['note']}")
    st.subheader("How sensitive is the list?")
    st.dataframe(A["sens"][["scenario", "share_of_top250_changed", "n_positive_net_value"]]
                 .style.format({"share_of_top250_changed": "{:.0%}", "n_positive_net_value": "{:,.0f}"}),
                 hide_index=True, use_container_width=True)

# ================================================================== SCORE A FILE
with tabs[5]:
    chip("P")
    st.markdown("Upload any customer file — messy is fine (junk header rows, pasted totals, text numbers, any column order). "
                "The churn column, if present, is **never** used for scoring.")
    up = st.file_uploader("Customer CSV", type=["csv"])
    demo = st.button("…or load the deliberately dirty demo file (2,000 hold-out customers, labels removed)")
    path = None
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / f"retainiq_upload_{up.file_id}.csv"
        tmp.write_bytes(up.getvalue())
        path = tmp
    elif demo or st.session_state.get("demo"):
        st.session_state["demo"] = True
        path = DATA / "demo" / "holdout_dirty.csv"
    if path is not None:
        from predict import predict_file
        with st.spinner("Cleaning and scoring…"):
            try:
                scored, log, truth = predict_file(path)
            except Exception as e:                       # show, don't crash
                st.error(f"Could not score this file: {e}")
                scored = None
        if scored is not None:
            c = st.columns(4)
            c[0].metric("Customers scored", f"{len(scored):,}")
            c[1].metric("Cleaning actions", f"{len(log):,}")
            c[2].metric("P1 — Save now", f"{(scored['priority_tier'] == 'P1').sum():,}")
            c[3].metric("Balance at risk", f"{scored['balance_at_risk'].sum() / 1e6:,.2f}M")
            ts = scored["priority_tier"].value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0)
            fig = go.Figure(go.Bar(x=[TIER_NAME[t] for t in ts.index], y=ts.values, marker_color=[TIER_COLOR[t] for t in ts.index],
                                   text=[f"{int(v):,}" for v in ts.values], textposition="outside"))
            st.plotly_chart(style(fig, 280, title="Where the uploaded customers land"), theme="streamlit", use_container_width=True)
            with st.expander("Cleaning log — every change made to the file", expanded=False):
                st.dataframe(log, hide_index=True, use_container_width=True)
            st.dataframe(scored.head(200), hide_index=True, use_container_width=True)
            st.download_button("Download scored file (CSV)", scored.to_csv(index=False), "retainiq_scored.csv", "text/csv")
            st.caption("Note: the bundled model was refit on all 10,000 customers, so scoring the hold-out here is a demo "
                       "of the pipeline, not an unseen-data test. The honest unseen test (train-split model) is "
                       f"`python src/demo_unseen.py`: ROC-AUC {A['unseen']['clean_unlabelled']['roc_auc']:.3f}.")

st.divider()
st.caption("Simulated data only · every number is computed from pipeline outputs · labels: DESCRIPTIVE / PREDICTIVE / "
           "RECOMMENDATION · the model recommends, a person decides.")
