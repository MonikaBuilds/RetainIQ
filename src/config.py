"""RetainIQ - central configuration.

Every business assumption lives here so it can be changed in one place and
sensitivity-tested. Anything tagged [ASSUMPTION] is a judgement call, not a fact
taken from the data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Bank_Data.csv"
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REPORTS = ROOT / "reports"
for _p in (DATA, OUT, FIG, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
HOLDOUT_FRAC = 0.20
CV_FOLDS = 5
CV_REPEATS = 2
N_BOOT = 1000
AUG_TARGET_TOTAL = 50_000          # real (10k) + synthetic rows = 50k

TARGET = "churned"
ID = "customer_id"

EXPECTED_COLUMNS = [
    "customer_id", "age", "gender", "region", "customer_segment", "tenure_years",
    "monthly_income", "avg_monthly_balance", "num_products", "has_credit_card",
    "has_loan", "has_investment_product", "digital_engagement_score",
    "branch_visits_last_year", "complaint_count_last_year", "credit_score",
    "relationship_manager_assigned", "nps_score", "last_transaction_days_ago",
    "churned",
]
NUMERIC_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in
                   ("customer_id", "gender", "region", "customer_segment")]
BINARY_COLUMNS = ["has_credit_card", "has_loan", "has_investment_product",
                  "relationship_manager_assigned", "churned"]
SEGMENTS = ["Mass Market", "Mass Affluent", "Affluent", "Private Banking"]
REGIONS = ["North", "South", "East", "West", "Metro"]
GENDERS = ["Male", "Female"]

# Validity ranges (domain limits) used by clean.py
VALID_RANGES = {
    "age": (18, 100),
    "credit_score": (300, 850),
    "digital_engagement_score": (0, 100),
    "nps_score": (-100, 100),
    "tenure_years": (0, 82),
    "num_products": (1, 10),
    "branch_visits_last_year": (0, 365),
    "complaint_count_last_year": (0, 50),
    "last_transaction_days_ago": (0, 3650),
    "monthly_income": (0, None),
    "avg_monthly_balance": (0, None),
}

# ---- Business value assumptions --------------------------------------------
# [ASSUMPTION] avg_monthly_balance is a stock (balance), in an unspecified
# currency unit. Annual value of keeping a balance = balance x net interest
# margin. The brief's suggestion of "x 12" would treat a stock as a monthly
# flow and overstate value 12-fold, so we do NOT multiply by 12.
NIM = 0.025                     # [ASSUMPTION] net interest margin on balances, 2.5%/yr
FEE_PER_PRODUCT = 50.0          # [ASSUMPTION] annual fee/cross-sell margin per product held
VALUE_HORIZON_YEARS = 3         # [ASSUMPTION] years of margin preserved when a customer is saved (sensitivity: 1, 5)
SAVE_RATE = 0.20               # [ASSUMPTION] share of truly at-risk customers an intervention saves
SAVE_RATE_RANGE = (0.10, 0.30)
CONTACT_COST = {                # [ASSUMPTION] cost per customer contacted, by channel
    "rm_call_offer": 150.0,     # P1 - relationship manager call + bespoke offer
    "nurture": 40.0,            # P2 - cross-sell / loyalty programme
    "digital_nudge": 5.0,       # P3 - automated app/email journey
    "monitor": 0.0,             # P4
}
DEFAULT_CONTACT_COST = 25.0     # [ASSUMPTION] blended cost used for threshold choice
WINSOR_PCT = 0.99               # balance winsorisation for sensitivity views

import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
