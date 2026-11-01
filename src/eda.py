"""RetainIQ - Stage 2: SEE. Descriptive evidence, hypothesis tests, clustering, figures.

All findings here come from the 10,000 REAL customers only (never synthetic rows).
Language rule: 'associated with', never 'causes'.
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

from config import DATA, FIG, OUT, SEED, TARGET
from features import BASE_FEATURES, INACTIVITY_FEATURES, NPS_FEATURES

warnings.filterwarnings("ignore")

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b8b7b1", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})
RESULTS = {}
PVALS = {}


# ------------------------------------------------------------------ helpers
def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return lo, hi


def rate_table(df, col, order=None):
    g = df.groupby(col, observed=True)
    t = pd.DataFrame({
        "customers": g.size(),
        "churners": g[TARGET].sum(),
        "total_balance": g["avg_monthly_balance"].sum(),
        "churned_balance": g.apply(lambda x: x.loc[x[TARGET] == 1, "avg_monthly_balance"].sum()),
    })
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    t["churn_rate"] = t["churners"] / t["customers"]
    ci = [wilson(k, n) for k, n in zip(t["churners"], t["customers"])]
    t["ci_low"], t["ci_high"] = zip(*ci)
    t["share_of_churners"] = t["churners"] / df[TARGET].sum()
    t["share_of_churned_balance"] = t["churned_balance"] / df.loc[df[TARGET] == 1, "avg_monthly_balance"].sum()
    return t


def cochran_armitage(cases, totals, scores=None):
    cases, totals = np.asarray(cases, float), np.asarray(totals, float)
    s = np.arange(len(cases)) if scores is None else np.asarray(scores, float)
    N, p = totals.sum(), cases.sum() / totals.sum()
    T = np.sum(s * (cases - totals * p))
    var = p * (1 - p) * (np.sum(totals * s ** 2) - np.sum(totals * s) ** 2 / N)
    z = T / np.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


def chi2_cramer(df, col):
    ct = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    return chi2, p, v


def odds_ratio(df, mask):
    a = ((mask) & (df[TARGET] == 1)).sum(); b = ((mask) & (df[TARGET] == 0)).sum()
    c = ((~mask) & (df[TARGET] == 1)).sum(); d = ((~mask) & (df[TARGET] == 0)).sum()
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)


def logit_or(model, term):
    ci = model.conf_int().loc[term]
    return float(np.exp(model.params[term])), float(np.exp(ci[0])), float(np.exp(ci[1])), float(model.pvalues[term])


def pct(x):
    return f"{x:.1%}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bar_rates(ax, t, labels=None, highlight=None, color=BLUE):
    x = np.arange(len(t))
    cols = [ORANGE if (highlight is not None and i in highlight) else color for i in range(len(t))]
    ax.bar(x, t["churn_rate"], width=0.62, color=cols, zorder=2)
    ax.errorbar(x, t["churn_rate"], yerr=[t["churn_rate"] - t["ci_low"], t["ci_high"] - t["churn_rate"]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=3, zorder=3)
    for i, r in enumerate(t["churn_rate"]):
        ax.text(i, t["ci_high"].iloc[i] + 0.008, f"{r:.1%}", ha="center", va="bottom", fontsize=8.5, color=INK)
    names = labels if labels is not None else [str(v) for v in t.index]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\nn={int(n):,}" for nm, n in zip(names, t["customers"])], fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(t["ci_high"]) * 1.2)
    ax.set_ylabel("Churn rate (95% Wilson CI)")


# ------------------------------------------------------------------ main
def main():
    df = pd.read_csv(DATA / "cleaned_bank_data.csv")
    n, churners = len(df), int(df[TARGET].sum())
    base = churners / n
    tot_bal = df["avg_monthly_balance"].sum()
    churn_bal = df.loc[df[TARGET] == 1, "avg_monthly_balance"].sum()
    RESULTS["overall"] = dict(customers=n, churners=churners, churn_rate=base,
                              churn_rate_ci=wilson(churners, n),
                              churned_balance_share=churn_bal / tot_bal,
                              total_balance=tot_bal, churned_balance=churn_bal,
                              accuracy_of_predict_nobody=1 - base)
    # winsorised view of the value insight
    cap = df["avg_monthly_balance"].quantile(0.99)
    wb = df["avg_monthly_balance"].clip(upper=cap)
    RESULTS["overall"]["churned_balance_share_winsor99"] = wb[df[TARGET] == 1].sum() / wb.sum()
    print(f"Overall churn {pct(base)} | churned balance share {pct(churn_bal / tot_bal)}")

    df["balance_quintile"] = pd.qcut(df["avg_monthly_balance"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    seg_order = ["Mass Market", "Mass Affluent", "Affluent", "Private Banking"]
    tables = {
        "segment": rate_table(df, "customer_segment", seg_order),
        "products": rate_table(df, "product_band", ["1", "2", "3", "4", "5+"]),
        "complaints": rate_table(df, "complaint_band", ["0", "1", "2", "3+"]),
        "complaints_raw": rate_table(df, "complaint_count_last_year"),
        "rm": rate_table(df, "relationship_manager_assigned"),
        "inactivity": rate_table(df, "inactivity_band", ["<=15", "16-45", "46-90", "90+"]),
        "balance_quintile": rate_table(df, "balance_quintile"),
        "tenure": rate_table(df, "tenure_band", ["<1", "1-3", "3-7", "7-15", "15+"]),
        "region": rate_table(df, "region"),
        "gender": rate_table(df, "gender"),
        "age_band": rate_table(df, "age_band", ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]),
        "low_engagement": rate_table(df, "low_engagement"),
    }
    pd.concat([t.reset_index().rename(columns={t.index.name or "index": "level"}).astype({"level": str})
               .assign(cut=k) for k, t in tables.items()], ignore_index=True) \
        .to_csv(OUT / "eda_tables.csv", index=False)
    for k, t in tables.items():
        RESULTS[f"table_{k}"] = {str(i): {c: (float(v) if np.isscalar(v) else v) for c, v in r.items()}
                                 for i, r in t.iterrows()}
    print(tables["complaints_raw"][["customers", "churn_rate"]])

    # ---------------- univariate AUC: which single field separates best?
    numeric = ["avg_monthly_balance", "monthly_income", "num_products", "tenure_years", "age",
               "digital_engagement_score", "branch_visits_last_year", "complaint_count_last_year",
               "credit_score", "nps_score", "last_transaction_days_ago", "relationship_manager_assigned",
               "segment_rank"]
    uni = []
    for c in numeric:
        a = roc_auc_score(df[TARGET], df[c])
        u, p = stats.mannwhitneyu(df.loc[df[TARGET] == 1, c], df.loc[df[TARGET] == 0, c])
        uni.append(dict(feature=c, auc=max(a, 1 - a), direction="higher->more churn" if a > 0.5 else "lower->more churn",
                        median_churned=df.loc[df[TARGET] == 1, c].median(),
                        median_stayed=df.loc[df[TARGET] == 0, c].median(),
                        rank_biserial=2 * a - 1, mw_p=p))
        PVALS[f"MW_{c}"] = p
    uni = pd.DataFrame(uni).sort_values("auc", ascending=False)
    uni.to_csv(OUT / "univariate_auc.csv", index=False)
    RESULTS["univariate_auc"] = uni.set_index("feature")["auc"].round(4).to_dict()
    print(uni[["feature", "auc", "direction"]])

    # ---------------- hypotheses
    H = {}
    t = tables["complaints"]
    z, p = cochran_armitage(t["churners"], t["customers"])
    orr = odds_ratio(df, df["complaint_count_last_year"] >= 2)
    H["H1_complaints_dose_response"] = dict(
        rates={k: float(v) for k, v in t["churn_rate"].items()}, ca_z=z, p=p,
        ratio_2_vs_0=float(t.loc["2", "churn_rate"] / t.loc["0", "churn_rate"]),
        or_2plus_vs_lt2=orr,
        note="3+ grouped (n=%d); raw complaints=4 n=%d, =5 n=%d not quoted" % (
            t.loc["3+", "customers"], (df.complaint_count_last_year == 4).sum(), (df.complaint_count_last_year == 5).sum()),
        supported=bool(p < 0.05 and z > 0))
    PVALS["H1"] = p

    rho, p = stats.spearmanr(df["avg_monthly_balance"], df[TARGET])
    m = smf.logit("churned ~ log_balance", df).fit(disp=0)
    H["H2_low_balance"] = dict(spearman=rho, p=p, or_per_log_unit=logit_or(m, "log_balance"),
                               quintile_rates={str(k): float(v) for k, v in tables["balance_quintile"]["churn_rate"].items()},
                               supported=bool(p < 0.05 and rho < 0))
    PVALS["H2"] = p

    chi2, p, v = chi2_cramer(df, "single_product")
    H["H3_single_product"] = dict(single_rate=float(df.loc[df.single_product == 1, TARGET].mean()),
                                  multi_rate=float(df.loc[df.single_product == 0, TARGET].mean()),
                                  odds_ratio=odds_ratio(df, df.single_product == 1), chi2=chi2, p=p, cramers_v=v,
                                  share_of_customers=float(df.single_product.mean()),
                                  share_of_churners=float(df.loc[df[TARGET] == 1, "single_product"].mean()),
                                  supported=bool(p < 0.05))
    PVALS["H3"] = p

    t = tables["inactivity"]
    z, p = cochran_armitage(t["churners"], t["customers"])
    H["H4_inactivity"] = dict(rates={k: float(v) for k, v in t["churn_rate"].items()}, ca_z=z, p=p,
                              univariate_auc=float(RESULTS["univariate_auc"]["last_transaction_days_ago"]),
                              rank_among_fields=int(list(uni.feature).index("last_transaction_days_ago") + 1),
                              ratio_90plus_vs_le15=float(t.loc["90+", "churn_rate"] / t.loc["<=15", "churn_rate"]),
                              supported=bool(p < 0.05),
                              note="strongest single *behavioural* field; see leakage caveat in train.py")
    PVALS["H4"] = p

    chi2, p, v = chi2_cramer(df, "low_engagement")
    by_seg = df.groupby(["customer_segment", "low_engagement"])[TARGET].mean().unstack()
    m0 = smf.logit("churned ~ digital_engagement_score + C(customer_segment)", df).fit(disp=0)
    m1 = smf.logit("churned ~ digital_engagement_score * C(customer_segment)", df).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf); p_int = stats.chi2.sf(lr, m1.df_model - m0.df_model)
    H["H5_low_engagement"] = dict(low_rate=float(df.loc[df.low_engagement == 1, TARGET].mean()),
                                  other_rate=float(df.loc[df.low_engagement == 0, TARGET].mean()), p=p, cramers_v=v,
                                  by_segment=by_seg.round(4).to_dict(orient="index"),
                                  interaction_lr_p=p_int, supported=bool(p < 0.05))
    PVALS["H5"] = p; PVALS["H5_interaction"] = p_int

    t = tables["tenure"]
    z, p = cochran_armitage(t["churners"], t["customers"])
    H["H6_early_tenure"] = dict(rates={k: float(v) for k, v in t["churn_rate"].items()}, ca_z=z, p=p,
                                supported=bool(p < 0.05 and z < 0 and t["churn_rate"].idxmax() == "<1"))
    PVALS["H6"] = p

    # H7 RM confounding
    rm_seg = df.groupby(["customer_segment", "relationship_manager_assigned"]).agg(
        n=(TARGET, "size"), churn=(TARGET, "mean")).unstack()
    rm_share = df.groupby("customer_segment")["relationship_manager_assigned"].mean()
    crude = smf.logit("churned ~ relationship_manager_assigned", df).fit(disp=0)
    adj = smf.logit("churned ~ relationship_manager_assigned + C(customer_segment) + log_balance + "
                    "tenure_years + num_products", df).fit(disp=0)
    H["H7_rm_confounding"] = dict(
        rm_share_by_segment=rm_share.round(3).to_dict(),
        median_balance_rm=float(df.loc[df.relationship_manager_assigned == 1, "avg_monthly_balance"].median()),
        median_balance_no_rm=float(df.loc[df.relationship_manager_assigned == 0, "avg_monthly_balance"].median()),
        churn_by_segment_rm={s: {"no_rm": float(rm_seg.loc[s, ("churn", 0)]), "rm": float(rm_seg.loc[s, ("churn", 1)]),
                                 "n_no_rm": int(rm_seg.loc[s, ("n", 0)]), "n_rm": int(rm_seg.loc[s, ("n", 1)])}
                             for s in seg_order},
        crude_or=logit_or(crude, "relationship_manager_assigned"),
        adjusted_or=logit_or(adj, "relationship_manager_assigned"),
        supported=bool(adj.pvalues["relationship_manager_assigned"] < 0.05 and adj.params["relationship_manager_assigned"] < 0),
        note="association only; RM is assigned to wealthy customers, cannot claim RM causes retention")
    PVALS["H7"] = float(adj.pvalues["relationship_manager_assigned"])

    # H8 interactions
    m8 = smf.logit("churned ~ has_complaint * inactive_45plus", df).fit(disp=0)
    df["negative_nps"] = (df.nps_score < 0).astype(int)
    m8b = smf.logit("churned ~ has_complaint * negative_nps", df).fit(disp=0)
    grid = df.groupby(["has_complaint", "inactive_45plus"])[TARGET].agg(["mean", "size"])
    grid2 = df.groupby(["has_complaint", "negative_nps"])[TARGET].agg(["mean", "size"])
    H["H8_interactions"] = dict(
        complaint_x_inactive={f"complaint={int(a)},inactive45={int(b)}": dict(rate=float(r["mean"]), n=int(r["size"]))
                              for (a, b), r in grid.iterrows()},
        complaint_x_inactive_interaction_or=logit_or(m8, "has_complaint:inactive_45plus"),
        complaint_x_negnps={f"complaint={int(a)},negNPS={int(b)}": dict(rate=float(r["mean"]), n=int(r["size"]))
                            for (a, b), r in grid2.iterrows()},
        complaint_x_negnps_interaction_or=logit_or(m8b, "has_complaint:negative_nps"),
        note="interaction OR on log-odds scale; ~1 means effects simply add on the log-odds scale")
    PVALS["H8a"] = logit_or(m8, "has_complaint:inactive_45plus")[3]
    PVALS["H8b"] = logit_or(m8b, "has_complaint:negative_nps")[3]

    H["H9_region_gender_flat"] = {}
    for c in ("region", "gender"):
        chi2, p, v = chi2_cramer(df, c)
        H["H9_region_gender_flat"][c] = dict(chi2=chi2, p=p, cramers_v=v,
                                             rates=df.groupby(c)[TARGET].mean().round(4).to_dict())
        PVALS[f"H9_{c}"] = p

    adj10 = smf.logit("churned ~ credit_score + log_income + age + C(customer_segment) + log_balance + num_products",
                      df).fit(disp=0)
    H["H10_weak_fields"] = {c: dict(univariate_auc=float(RESULTS["univariate_auc"][c0]),
                                    adjusted=logit_or(adj10, c))
                            for c, c0 in (("credit_score", "credit_score"), ("log_income", "monthly_income"), ("age", "age"))}
    for c in ("credit_score", "log_income", "age"):
        PVALS[f"H10_{c}"] = logit_or(adj10, c)[3]

    # H11 competitor-switch-risk proxy (NOT confirmed switching)
    df["switch_risk_signals"] = (df.low_engagement + df.inactive_45plus + df.single_product + df.has_complaint)
    t11 = rate_table(df, "switch_risk_signals")
    z, p = cochran_armitage(t11["churners"], t11["customers"])
    H["H11_competitor_switch_proxy"] = dict(
        definition="count of: low digital engagement, 46+ days inactive, single product, any complaint",
        rates={str(int(k)): dict(rate=float(r.churn_rate), n=int(r.customers)) for k, r in t11.iterrows()},
        ca_z=z, p=p, limitation="no competitor field in data; this is a risk *profile*, not confirmed switching")
    PVALS["H11"] = p

    keys = list(PVALS); rej, padj, _, _ = multipletests([PVALS[k] for k in keys], method="fdr_bh")
    RESULTS["bh_adjusted_p"] = {k: float(v) for k, v in zip(keys, padj)}
    RESULTS["hypotheses"] = H

    # ---------------- volume vs value problem
    ch = df[df[TARGET] == 1]
    top_q = df["avg_monthly_balance"].quantile(0.80)
    RESULTS["volume_vs_value"] = dict(
        mass_market_share_of_churners=float((ch.customer_segment == "Mass Market").mean()),
        mass_market_share_of_churned_balance=float(ch.loc[ch.customer_segment == "Mass Market", "avg_monthly_balance"].sum() / ch.avg_monthly_balance.sum()),
        top_quintile_balance_share_of_churners=float((ch.avg_monthly_balance >= top_q).mean()),
        top_quintile_balance_share_of_churned_balance=float(ch.loc[ch.avg_monthly_balance >= top_q, "avg_monthly_balance"].sum() / ch.avg_monthly_balance.sum()),
        n_high_balance_leavers=int((ch.avg_monthly_balance >= top_q).sum()),
        median_balance_churned=float(ch.avg_monthly_balance.median()),
        median_balance_stayed=float(df.loc[df[TARGET] == 0, "avg_monthly_balance"].median()))

    # ---------------- multicollinearity
    X = df[BASE_FEATURES + INACTIVITY_FEATURES + NPS_FEATURES].astype(float)
    Xc = sm.add_constant(X)
    vif = pd.DataFrame({"feature": X.columns,
                        "vif": [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]})
    vif.sort_values("vif", ascending=False).to_csv(OUT / "vif.csv", index=False)
    RESULTS["vif_top"] = vif.sort_values("vif", ascending=False).head(8).set_index("feature")["vif"].round(2).to_dict()

    clusters = run_clustering(df)
    make_figures(df, tables, clusters)
    json.dump(RESULTS, open(OUT / "eda_results.json", "w"), indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o))
    print(json.dumps({k: RESULTS[k] for k in ("overall", "volume_vs_value")}, indent=2, default=float))
    print("\nPASS: all EDA on real rows only (n=%d)" % len(df))


# ------------------------------------------------------------------ clustering
CLUSTER_VARS = ["log_balance", "log_income", "num_products", "digital_engagement_score",
                "branch_visits_last_year", "complaints_grp", "tenure_years", "log_days_inactive",
                "nps_score", "relationship_manager_assigned"]


def name_cluster(prof, overall, sd):
    z = (prof - overall) / sd
    parts = []
    wealth = prof["segment_rank"]
    parts.append("Wealthy" if wealth >= 1.8 else ("Comfortable" if wealth >= 0.8 else "Mass Market"))
    tags = []
    if z["num_products"] <= -0.5: tags.append(("Single-Product", abs(z["num_products"])))
    if z["num_products"] >= 0.5: tags.append(("Multi-Product", z["num_products"]))
    if z["complaints_grp"] >= 0.5: tags.append(("Disgruntled", z["complaints_grp"]))
    if z["last_transaction_days_ago"] >= 0.5: tags.append(("Quiet", z["last_transaction_days_ago"]))
    if z["digital_engagement_score"] >= 0.5: tags.append(("Digital", z["digital_engagement_score"]))
    if z["digital_engagement_score"] <= -0.5: tags.append(("Low-Digital", abs(z["digital_engagement_score"])))
    if z["tenure_years"] >= 0.6: tags.append(("Veteran", z["tenure_years"]))
    if z["branch_visits_last_year"] >= 0.6: tags.append(("Branch-Reliant", z["branch_visits_last_year"]))
    if z["nps_score"] <= -0.6: tags.append(("Unhappy", abs(z["nps_score"])))
    tags = [t for t, _ in sorted(tags, key=lambda x: -x[1])[:2]]
    return " ".join(tags + [parts[0]]) if tags else f"Steady {parts[0]}"


def run_clustering(df):
    X = StandardScaler().fit_transform(df[CLUSTER_VARS])
    rng = np.random.RandomState(SEED)
    samp = rng.choice(len(X), 4000, replace=False)
    sel = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        gm = GaussianMixture(k, random_state=SEED, n_init=2).fit(X)
        sel.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X[samp], km.labels_[samp]),
                        gmm_bic=gm.bic(X)))
    sel = pd.DataFrame(sel)

    def stability(k):
        base_lab = KMeans(k, n_init=20, random_state=SEED).fit(X).labels_
        out = []
        for b in range(10):
            idx = np.random.RandomState(SEED + b).choice(len(X), len(X), replace=True)
            kb = KMeans(k, n_init=10, random_state=SEED + b).fit(X[idx])
            out.append(adjusted_rand_score(base_lab, kb.predict(X)))
        return out

    # Rule: among k in 3..7 whose silhouette is within 2% of the best, take the
    # largest k that is bootstrap-stable (mean ARI >= 0.8): more actionable
    # personas without sacrificing measurable separation or stability.
    cand = sel[(sel.k >= 3) & (sel.k <= 7)]
    near = cand[cand.silhouette >= 0.98 * cand.silhouette.max()].sort_values("k", ascending=False)
    sel["bootstrap_ari"] = np.nan
    k_best, aris = None, None
    for k in near.k:
        a = stability(int(k))
        sel.loc[sel.k == k, "bootstrap_ari"] = np.mean(a)
        if np.mean(a) >= 0.8:
            k_best, aris = int(k), a
            break
    if k_best is None:
        k_best = int(cand.loc[cand.silhouette.idxmax(), "k"]); aris = stability(k_best)
    km = KMeans(k_best, n_init=20, random_state=SEED).fit(X)
    labels = km.labels_
    gm = GaussianMixture(k_best, random_state=SEED, n_init=3).fit(X)
    rules = df["customer_segment"] + "|" + df["single_product"].map({1: "single", 0: "multi"})
    sel.to_csv(OUT / "cluster_selection.csv", index=False)
    df["cluster"] = labels
    prof_cols = ["avg_monthly_balance", "segment_rank", "num_products", "digital_engagement_score",
                 "branch_visits_last_year", "complaints_grp", "tenure_years", "last_transaction_days_ago",
                 "nps_score", "relationship_manager_assigned"]
    overall, sd = df[prof_cols].mean(), df[prof_cols].std()
    prof = df.groupby("cluster")[prof_cols].mean()
    prof["median_balance"] = df.groupby("cluster")["avg_monthly_balance"].median()
    prof["size"] = df.groupby("cluster").size()
    prof["churn_rate"] = df.groupby("cluster")[TARGET].mean()
    prof["share_churned_balance"] = df[df[TARGET] == 1].groupby("cluster")["avg_monthly_balance"].sum() / \
        df.loc[df[TARGET] == 1, "avg_monthly_balance"].sum()
    names = {c: name_cluster(prof.loc[c], overall, sd) for c in prof.index}
    seen = {}
    for c in sorted(names, key=lambda c: -prof.loc[c, "size"]):
        nm = names[c]; seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1: names[c] = f"{nm} ({seen[nm]})"
    prof["persona"] = prof.index.map(names)
    prof["play"] = prof.apply(cluster_play, axis=1)
    prof.to_csv(OUT / "cluster_profiles.csv")
    df["persona"] = df["cluster"].map(names)
    df[["customer_id", "cluster", "persona"]].to_csv(OUT / "cluster_assignments.csv", index=False)
    RESULTS["clustering"] = dict(
        k=k_best, selection=sel.round(4).to_dict(orient="records"),
        bootstrap_ari_mean=float(np.mean(aris)), bootstrap_ari_sd=float(np.std(aris)),
        ari_kmeans_vs_gmm=float(adjusted_rand_score(labels, gm.predict(X))),
        ari_kmeans_vs_rules=float(adjusted_rand_score(labels, rules)),
        profiles=prof.round(3).reset_index().to_dict(orient="records"),
        note="churned not used as a clustering input; used only to evaluate clusters")
    print(prof[["persona", "size", "churn_rate", "median_balance", "num_products", "complaints_grp",
                "last_transaction_days_ago", "share_churned_balance"]])
    return prof


def cluster_play(r):
    if r["churn_rate"] >= 0.15 and r["complaints_grp"] >= 1:
        return "Complaint recovery: 48-hour resolution SLA + proactive follow-up call"
    if r["churn_rate"] >= 0.2 and r["median_balance"] < 10000:
        return "P3 low-cost automation: second-product bundle + re-engagement nudge"
    if r["churn_rate"] >= 0.15:
        return "Targeted save: complaint follow-up / re-engagement, RM call if balance high"
    if r["segment_rank"] >= 1.8:
        return "P2 nurture & deepen: protect relationship, wealth cross-sell"
    return "Monitor; light-touch loyalty"


# ------------------------------------------------------------------ figures
def make_figures(df, T, prof):
    seg = T["segment"]
    # 1 segment
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    bar_rates(axes[0], seg, labels=["Mass\nMarket", "Mass\nAffluent", "Affluent", "Private\nBanking"], highlight=[0])
    axes[0].set_title("Churn rate by segment")
    axes[1].barh(np.arange(4)[::-1], seg["churners"], color=[ORANGE, BLUE, BLUE, BLUE], height=0.6)
    for i, v in enumerate(seg["churners"]):
        axes[1].text(v + 15, 3 - i, f"{int(v):,} ({seg['share_of_churners'].iloc[i]:.0%})", va="center", fontsize=8)
    axes[1].set_yticks(np.arange(4)[::-1]); axes[1].set_yticklabels(seg.index)
    axes[1].set_title("Number of churners"); axes[1].set_xlim(0, seg["churners"].max() * 1.35)
    axes[1].grid(axis="y", visible=False)
    r = seg.loc["Mass Market", "churn_rate"] / seg.loc["Private Banking", "churn_rate"]
    fig.suptitle(f"Mass Market churns at {r:.1f}x the Private Banking rate and supplies "
                 f"{seg.loc['Mass Market', 'share_of_churners']:.0%} of all churners", fontweight="bold", fontsize=11, x=0.02, ha="left")
    save(fig, "01_churn_by_segment.png")

    # 2 complaints dose-response
    t = T["complaints"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(t))
    ax.fill_between(x, t["ci_low"], t["ci_high"], color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, t["churn_rate"], color=BLUE, lw=2, marker="o", ms=7)
    for i, (rr, nn) in enumerate(zip(t["churn_rate"], t["customers"])):
        ax.annotate(f"{rr:.1%}\n(n={nn:,})", (i, rr), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["0", "1", "2", "3+"]); ax.set_xlabel("Complaints in the last year")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, 0)); ax.set_ylabel("Churn rate (95% CI band)")
    ax.set_ylim(0, t["ci_high"].max() * 1.3)
    ax.set_title(f"Customers with 2 complaints churn at {t.loc['2', 'churn_rate'] / t.loc['0', 'churn_rate']:.1f}x "
                 f"the rate of customers with none\n(3+ grouped: wide CI, n={t.loc['3+', 'customers']})", loc="left")
    save(fig, "02_complaints_dose_response.png")

    # 3 balance quintile: rate vs share of churned balance (two panels, no dual axis)
    t = T["balance_quintile"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bar_rates(axes[0], t, labels=["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"], highlight=[0])
    axes[0].set_title("Churn rate by balance quintile")
    axes[1].bar(np.arange(5), t["share_of_churners"], width=0.36, color=BLUE, label="Share of churners", zorder=2)
    axes[1].bar(np.arange(5) + 0.38, t["share_of_churned_balance"], width=0.36, color=ORANGE, label="Share of churned balance", zorder=2)
    for i in range(5):
        axes[1].text(i, t["share_of_churners"].iloc[i] + 0.01, f"{t['share_of_churners'].iloc[i]:.0%}", ha="center", fontsize=7.5)
        axes[1].text(i + 0.38, t["share_of_churned_balance"].iloc[i] + 0.01, f"{t['share_of_churned_balance'].iloc[i]:.0%}", ha="center", fontsize=7.5)
    axes[1].set_xticks(np.arange(5) + 0.19); axes[1].set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, 0)); axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Where the churners vs where the money is")
    fig.suptitle(f"The lowest-balance quintile churns most, but the top quintile holds "
                 f"{t.loc['Q5 highest', 'share_of_churned_balance']:.0%} of the balance that left",
                 fontweight="bold", fontsize=11, x=0.02, ha="left")
    save(fig, "03_balance_quintile_rate_vs_value.png")

    # 4 products
    t = T["products"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bar_rates(ax, t, highlight=[0]); ax.set_xlabel("Products held")
    ax.set_title(f"Single-product customers churn at {t.loc['1', 'churn_rate']:.0%}, "
                 f"{t.loc['1', 'churn_rate'] / t.loc['2', 'churn_rate']:.1f}x two-product customers", loc="left")
    save(fig, "04_products_vs_churn.png")

    # 5 inactivity
    t = T["inactivity"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bar_rates(ax, t, labels=["<=15 days", "16-45", "46-90", "90+ days"], highlight=[3])
    ax.set_xlabel("Days since last transaction")
    ax.set_title(f"The Last 90 Days: customers silent for 90+ days churn at {t.loc['90+', 'churn_rate']:.0%}\n"
                 "(may partly be a symptom of leaving - see leakage test)", loc="left")
    save(fig, "05_inactivity_silent_exits.png")

    # 6 tenure
    t = T["tenure"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(t))
    ax.fill_between(x, t["ci_low"], t["ci_high"], color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, t["churn_rate"], color=BLUE, lw=2, marker="o", ms=7)
    for i, (rr, nn) in enumerate(zip(t["churn_rate"], t["customers"])):
        ax.annotate(f"{rr:.1%}\nn={nn:,}", (i, rr), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["<1 yr", "1-3", "3-7", "7-15", "15+ yrs"]); ax.set_xlabel("Tenure")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, 0)); ax.set_ylim(0, t["ci_high"].max() * 1.3)
    ax.set_ylabel("Churn rate (95% CI band)")
    ax.set_title(f"Churn falls steadily with tenure: {t.loc['<1', 'churn_rate']:.0%} in year one vs "
                 f"{t.loc['15+', 'churn_rate']:.0%} after 15 years", loc="left")
    save(fig, "06_tenure_curve.png")

    # 7 RM faceted by segment (confounding chart)
    segs = ["Mass Market", "Mass Affluent", "Affluent", "Private Banking"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.36
    for j, (rm, col, lab) in enumerate(((0, GREY, "No RM"), (1, BLUE, "RM assigned"))):
        rates, los, his, ns = [], [], [], []
        for s in segs:
            sub = df[(df.customer_segment == s) & (df.relationship_manager_assigned == rm)]
            k, nn = sub[TARGET].sum(), len(sub)
            rates.append(k / nn if nn else np.nan); lo, hi = wilson(k, nn) if nn else (np.nan, np.nan)
            los.append(lo); his.append(hi); ns.append(nn)
        xs = np.arange(4) + (j - 0.5) * w
        ax.bar(xs, rates, w * 0.95, color=col, label=lab, zorder=2)
        ax.errorbar(xs, rates, yerr=[np.array(rates) - los, np.array(his) - rates], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        for xx, rr, nn, hh in zip(xs, rates, ns, his):
            ax.text(xx, hh + 0.01, f"{rr:.1%}\nn={nn:,}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(segs)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, 0)); ax.legend(frameon=False)
    ax.set_ylim(0, 0.32); ax.set_ylabel("Churn rate (95% CI)")
    h7 = RESULTS["hypotheses"]["H7_rm_confounding"]
    ax.set_title(f"RM customers churn less in every segment, but RMs go to wealthy customers:\n"
                 f"odds ratio {h7['crude_or'][0]:.2f} raw vs {h7['adjusted_or'][0]:.2f} after adjusting for segment, "
                 "balance, tenure, products (association, not cause)", loc="left")
    save(fig, "07_rm_confounding_by_segment.png")

    # 8 heatmap segment x products
    pv = df.pivot_table(index="customer_segment", columns="product_band", values=TARGET, aggfunc="mean").reindex(segs)
    cnt = df.pivot_table(index="customer_segment", columns="product_band", values=TARGET, aggfunc="size").reindex(segs)
    pv = pv.where(cnt >= 30)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(pv.values, cmap="Blues", vmin=0, vmax=np.nanmax(pv.values))
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v, c = pv.values[i, j], cnt.values[i, j]
            txt = f"{v:.0%}\nn={int(c):,}" if not np.isnan(v) else (f"n={int(c)}\n(<30)" if not np.isnan(c) else "-")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if (not np.isnan(v) and v > 0.25) else INK)
    ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns); ax.set_xlabel("Products held")
    ax.set_yticks(range(4)); ax.set_yticklabels(segs); ax.grid(False)
    fig.colorbar(im, ax=ax, format=matplotlib.ticker.PercentFormatter(1.0, 0), label="Churn rate", shrink=0.8)
    ax.set_title(f"Single-product customers are the hot spot in both mass segments "
                 f"({pv.loc['Mass Market', '1']:.0%} Mass Market, {pv.loc['Mass Affluent', '1']:.0%} Mass Affluent)\n"
                 "(cells with n<30 suppressed)", loc="left")
    save(fig, "08_heatmap_segment_x_products.png")

    # 9 cluster profiles: z-score heatmap + churn bar (small multiples)
    cols = ["avg_monthly_balance", "num_products", "digital_engagement_score", "branch_visits_last_year",
            "complaints_grp", "tenure_years", "last_transaction_days_ago", "nps_score", "relationship_manager_assigned"]
    p = prof.sort_values("churn_rate", ascending=False)
    z = (p[cols] - df[cols].mean()) / df[cols].std()
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(p) + 1.8), gridspec_kw={"width_ratios": [2.6, 1]})
    im = axes[0].imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    axes[0].set_xticks(range(len(cols))); axes[0].set_xticklabels(["balance", "products", "digital", "branch", "complaints",
                                                                    "tenure", "days inactive", "NPS", "RM"], rotation=30, ha="right")
    axes[0].set_yticks(range(len(p))); axes[0].set_yticklabels([f"{nm} (n={s:,})" for nm, s in zip(p.persona, p["size"])])
    axes[0].grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            axes[0].text(j, i, f"{z.values[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(z.values[i, j]) > 0.9 else INK)
    fig.colorbar(im, ax=axes[0], label="vs average (SD units)", shrink=0.8)
    axes[1].barh(np.arange(len(p)), p.churn_rate, color=[ORANGE if r > RESULTS["overall"]["churn_rate"] else BLUE for r in p.churn_rate], height=0.6)
    axes[1].invert_yaxis(); axes[1].set_yticks([]); axes[1].axvline(RESULTS["overall"]["churn_rate"], color=INK, lw=0.8, ls="--")
    for i, rr in enumerate(p.churn_rate):
        axes[1].text(rr + 0.005, i, f"{rr:.0%}", va="center", fontsize=8)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, 0)); axes[1].set_title("Churn rate (dashed = average)")
    fig.suptitle(f"{len(p)} customer personas (K-Means, churn NOT used to build them): churn ranges "
                 f"{p.churn_rate.min():.0%} to {p.churn_rate.max():.0%}", fontweight="bold", fontsize=11, x=0.02, ha="left")
    save(fig, "09_cluster_personas.png")

    # 10 correlation matrix
    cc = ["churned", "segment_rank", "log_balance", "log_income", "relationship_manager_assigned", "num_products",
          "has_investment_product", "tenure_years", "complaint_count_last_year", "digital_engagement_score",
          "last_transaction_days_ago", "nps_score", "credit_score", "age", "branch_visits_last_year"]
    corr = df[cc].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    short = ["CHURNED", "segment", "balance", "income", "RM", "products", "investment", "tenure", "complaints",
             "digital", "days inactive", "NPS", "credit", "age", "branch visits"]
    ax.set_xticks(range(len(cc))); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(len(cc))); ax.set_yticklabels(short); ax.grid(False)
    for i in range(len(cc)):
        for j in range(len(cc)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.values[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman correlation")
    ax.set_title("Segment, balance, income, RM and products move together (a wealth block):\n"
                 "do not read their individual model weights as independent drivers", loc="left")
    save(fig, "10_correlation_matrix.png")


if __name__ == "__main__":
    main()
