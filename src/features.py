"""RetainIQ - feature engineering (row-wise, deterministic, no label used).

Each feature has a one-line business rationale. Shared by train.py and predict.py.
"""
import numpy as np
import pandas as pd

SEG_RANK = {"Mass Market": 0, "Mass Affluent": 1, "Affluent": 2, "Private Banking": 3}

FEATURE_RATIONALE = {
    "log_balance": "balance is heavily right-skewed (median ~6.9k, max ~1.2m); log tames the tail",
    "log_income": "income skewed and floored at 1,500",
    "balance_to_income": "how much of their money the customer keeps here (share-of-wallet proxy)",
    "product_flag_count": "count of card/loan/investment products held",
    "products_per_tenure": "speed of relationship deepening",
    "single_product": "one-product customers have little switching cost",
    "complaints_grp": "complaints capped at 3+ (cells above 3 have n<10)",
    "has_complaint": "any complaint in last year",
    "log_days_inactive": "days since last transaction, log-scaled (diminishing marginal signal)",
    "inactive_45plus": "dormancy flag (46+ days without a transaction)",
    "low_engagement": "bottom-quartile digital engagement (cut-off fixed from data, no label)",
    "rm_eligible_unassigned": "balance typical of RM customers but no RM assigned",
    "segment_rank": "ordinal wealth segment",
}


def add_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = df.copy()
    bal = d["avg_monthly_balance"]
    inc = d["monthly_income"]
    d["log_balance"] = np.log1p(bal.clip(lower=0))
    d["log_income"] = np.log1p(inc.clip(lower=0))
    d["balance_to_income"] = bal / inc.replace(0, np.nan)
    d["product_flag_count"] = d[["has_credit_card", "has_loan", "has_investment_product"]].sum(axis=1, min_count=1)
    d["products_per_tenure"] = d["num_products"] / (d["tenure_years"] + 1)
    d["single_product"] = (d["num_products"] == 1).astype(float).where(d["num_products"].notna())
    d["complaints_grp"] = d["complaint_count_last_year"].clip(upper=3)
    d["has_complaint"] = (d["complaint_count_last_year"] > 0).astype(float).where(
        d["complaint_count_last_year"].notna())
    days = d["last_transaction_days_ago"]
    d["log_days_inactive"] = np.log1p(days.clip(lower=0))
    d["inactive_45plus"] = (days > 45).astype(float).where(days.notna())
    d["inactivity_band"] = pd.cut(days, [-np.inf, 15, 45, 90, np.inf],
                                  labels=["<=15", "16-45", "46-90", "90+"]).astype(str)
    d["low_engagement"] = (d["digital_engagement_score"] <= params["low_engagement_cutoff"]).astype(float).where(
        d["digital_engagement_score"].notna())
    d["rm_eligible_unassigned"] = ((bal >= params["rm_eligible_balance"]) &
                                   (d["relationship_manager_assigned"] == 0)).astype(float)
    d["segment_rank"] = d["customer_segment"].map(SEG_RANK)
    d["age_band"] = pd.cut(d["age"], [0, 25, 35, 45, 55, 65, 200],
                           labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]).astype(str)
    d["complaint_band"] = d["complaints_grp"].map({0: "0", 1: "1", 2: "2", 3: "3+"}).fillna("NA")
    d["product_band"] = d["num_products"].clip(upper=5).map(
        {1: "1", 2: "2", 3: "3", 4: "4", 5: "5+"}).fillna("NA")
    return d


# ---------------------------------------------------------------- model inputs
# Protected / no-signal attributes (gender, age, region) are deliberately NOT
# model inputs; they are used only in the fairness audit. [ASSUMPTION]
BASE_FEATURES = [
    "segment_rank", "tenure_years", "log_income", "log_balance", "balance_to_income",
    "num_products", "single_product", "has_credit_card", "has_loan", "has_investment_product",
    # product_flag_count excluded: it is the exact sum of the 3 product flags (VIF = inf)
    "products_per_tenure", "digital_engagement_score", "low_engagement",
    "branch_visits_last_year", "complaints_grp", "credit_score",
    "relationship_manager_assigned", "rm_eligible_unassigned",
    "flag_product_mismatch", "flag_tenure_gt_adult_years",
]
INACTIVITY_FEATURES = ["last_transaction_days_ago", "log_days_inactive", "inactive_45plus"]
NPS_FEATURES = ["nps_score"]

VARIANTS = {
    "A_full": BASE_FEATURES + INACTIVITY_FEATURES + NPS_FEATURES,
    "B_no_inactivity": BASE_FEATURES + NPS_FEATURES,
    "C_early_warning": BASE_FEATURES,
}
# guard against silently losing a feature (e.g. a stray comment in the list above)
assert len(BASE_FEATURES) == 20 and "digital_engagement_score" in BASE_FEATURES, BASE_FEATURES

PLAIN_ENGLISH = {
    "segment_rank": "wealth segment",
    "tenure_years": "tenure",
    "log_income": "income",
    "log_balance": "balance",
    "balance_to_income": "balance relative to income",
    "num_products": "number of products",
    "single_product": "holds only one product",
    "has_credit_card": "credit card",
    "has_loan": "loan",
    "has_investment_product": "investment product",
    "product_flag_count": "card/loan/investment count",
    "products_per_tenure": "products per year of tenure",
    "digital_engagement_score": "digital engagement",
    "low_engagement": "low digital engagement",
    "branch_visits_last_year": "branch visits",
    "complaints_grp": "complaints last year",
    "credit_score": "credit score",
    "relationship_manager_assigned": "relationship manager",
    "rm_eligible_unassigned": "high balance but no RM",
    "flag_product_mismatch": "product data inconsistency",
    "flag_tenure_gt_adult_years": "tenure/age inconsistency",
    "last_transaction_days_ago": "days since last transaction",
    "log_days_inactive": "days since last transaction",
    "inactive_45plus": "dormant 45+ days",
    "nps_score": "NPS",
}
