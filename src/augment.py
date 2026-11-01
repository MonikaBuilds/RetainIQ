"""RetainIQ - Stage 0: defensible 10k -> 50k augmentation (TRAINING SPLIT ONLY).

Method: class-conditional SMOTE-NC-style interpolation.
  * a parent row is drawn from the real TRAINING split, stratified by `churned`
    so the 15.8% churn rate is preserved;
  * one of its 5 nearest same-class neighbours is chosen;
  * continuous fields are interpolated (skewed fields in log space);
  * the "product/relationship block" (segment, num_products, card, loan,
    investment, RM) and other categoricals are copied whole from ONE donor, so
    logical rules (flags vs num_products, RM vs segment) are inherited, not invented.
Why this and not CTGAN/copula: no extra dependency, every synthetic row is
traceable to two real parents, and convex interpolation cannot create values
outside the observed range.

Synthetic rows NEVER enter the hold-out and NEVER feed descriptive statistics.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from config import EXPECTED_COLUMNS, ID, SEED, TARGET

LOG_FIELDS = ["monthly_income", "avg_monthly_balance", "last_transaction_days_ago"]
LIN_FIELDS = ["age", "tenure_years", "digital_engagement_score", "credit_score", "nps_score",
              "branch_visits_last_year", "complaint_count_last_year"]
INT_FIELDS = ["age", "branch_visits_last_year", "complaint_count_last_year", "credit_score",
              "nps_score", "last_transaction_days_ago", "monthly_income", "avg_monthly_balance"]
BLOCK = ["customer_segment", "num_products", "has_credit_card", "has_loan",
         "has_investment_product", "relationship_manager_assigned"]
OTHER_CATS = ["gender", "region"]


def _space(d):
    seg = d["customer_segment"].map({"Mass Market": 0, "Mass Affluent": 1, "Affluent": 2, "Private Banking": 3})
    m = np.column_stack([np.log1p(d[c]) for c in LOG_FIELDS] + [d[c] for c in LIN_FIELDS] +
                        [d["num_products"], seg])
    return StandardScaler().fit_transform(m)


def synthesize(train_raw: pd.DataFrame, n_new: int, seed=SEED, k=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    rate = train_raw[TARGET].mean()
    n1 = int(round(n_new * rate)); counts = {1: n1, 0: n_new - n1}
    for cls, n in counts.items():
        part = train_raw[train_raw[TARGET] == cls].reset_index(drop=True)
        Z = _space(part)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        _, idx = nn.kneighbors(Z)
        p = rng.integers(0, len(part), n)
        q = idx[p, rng.integers(1, k + 1, n)]
        A, B = part.iloc[p].reset_index(drop=True), part.iloc[q].reset_index(drop=True)
        lam = rng.random(n)
        s = pd.DataFrame(index=range(n))
        for c in LOG_FIELDS:
            s[c] = np.expm1(np.log1p(A[c]) + lam * (np.log1p(B[c]) - np.log1p(A[c])))
        for c in LIN_FIELDS:
            s[c] = A[c] + lam * (B[c] - A[c])
        donor_b = rng.random(n) < 0.5
        for c in BLOCK + OTHER_CATS:
            s[c] = np.where(donor_b, B[c], A[c])
        s[TARGET] = cls
        s["source_customer_id"] = A[ID].values
        s["source_neighbor_id"] = B[ID].values
        out.append(s)
    syn = pd.concat(out, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    for c in INT_FIELDS:
        syn[c] = syn[c].round()
    syn["tenure_years"] = syn["tenure_years"].round(1)
    syn["digital_engagement_score"] = syn["digital_engagement_score"].round(1)
    syn[ID] = [f"SYN{i:06d}" for i in range(len(syn))]
    syn["data_origin"] = "synthetic"
    return syn[EXPECTED_COLUMNS + ["data_origin", "source_customer_id", "source_neighbor_id"]]


def validate(real: pd.DataFrame, syn: pd.DataFrame, seed=SEED) -> dict:
    num = [c for c in EXPECTED_COLUMNS if c not in (ID, "gender", "region", "customer_segment", TARGET)]
    ks = {c: float(stats.ks_2samp(real[c], syn[c]).statistic) for c in num}
    chi = {}
    for c in ("gender", "region", "customer_segment"):
        ct = pd.concat([real[c].value_counts(normalize=True), syn[c].value_counts(normalize=True)], axis=1).fillna(0)
        chi[c] = float((ct.iloc[:, 0] - ct.iloc[:, 1]).abs().max())
    cr, cs = real[num + [TARGET]].corr(), syn[num + [TARGET]].corr()
    diff = (cr - cs).abs().values[np.triu_indices(len(cr), 1)]
    seg = pd.DataFrame({"real": real.groupby("customer_segment")[TARGET].mean(),
                        "synthetic": syn.groupby("customer_segment")[TARGET].mean()})
    # discriminator: can a model tell real from synthetic?
    m = min(len(real), len(syn))
    rs, ss = real.sample(m, random_state=seed), syn.sample(m, random_state=seed)
    X = pd.concat([rs, ss])[num + [TARGET]].copy()
    for c in ("gender", "region", "customer_segment"):
        X[c] = pd.concat([rs, ss])[c].astype("category").cat.codes.values
    y = np.r_[np.zeros(m), np.ones(m)]
    auc = cross_val_score(LGBMClassifier(n_estimators=200, verbose=-1, random_state=seed), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc")
    # memorisation: distance synthetic->nearest real vs real(half B)->real(half A)
    Zr = _space(real); half = len(Zr) // 2
    nnA = NearestNeighbors(n_neighbors=1).fit(Zr[:half])
    d_rr = nnA.kneighbors(Zr[half:])[0].ravel()
    both = pd.concat([real, syn.sample(min(len(syn), 5000), random_state=seed)])
    Zb = _space(both)
    nnR = NearestNeighbors(n_neighbors=1).fit(Zb[:len(real)])
    d_sr = nnR.kneighbors(Zb[len(real):])[0].ravel()
    return dict(
        ks_statistic=ks, ks_max=max(ks.values()), ks_mean=float(np.mean(list(ks.values()))),
        categorical_max_share_diff=chi,
        corr_abs_diff_max=float(diff.max()), corr_abs_diff_mean=float(diff.mean()),
        churn_rate_real=float(real[TARGET].mean()), churn_rate_synthetic=float(syn[TARGET].mean()),
        churn_by_segment=seg.round(4).to_dict(orient="index"),
        real_vs_synthetic_auc=float(auc.mean()), real_vs_synthetic_auc_sd=float(auc.std()),
        nn_distance_real_to_real_median=float(np.median(d_rr)),
        nn_distance_synth_to_real_median=float(np.median(d_sr)),
        exact_copies=int((d_sr < 1e-9).sum()),
        rule_flags_gt_products_real=float((real[["has_credit_card", "has_loan", "has_investment_product"]].sum(1) > real.num_products).mean()),
        rule_flags_gt_products_synth=float((syn[["has_credit_card", "has_loan", "has_investment_product"]].sum(1) > syn.num_products).mean()),
        rule_tenure_gt_age_synth=float((syn.tenure_years > syn.age).mean()),
        rule_tenure_gt_age_real=float((real.tenure_years > real.age).mean()),
    )
