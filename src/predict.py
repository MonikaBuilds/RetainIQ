"""RetainIQ - score ANY CSV with the challenge schema.

    python src/predict.py --input new_customers.csv --output scored.csv

The file passes through the SAME cleaning rules as training (junk header rows,
pasted summary rows, text numbers, column order, missing / out-of-range values).
A `churned` column, if present, is ignored for scoring.

Output columns: customer_id, churn_probability, risk_tier, priority_tier,
balance_at_risk, net_retention_value, priority_score, priority_rank,
top_reasons, recommended_action.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import CleaningLog, clean_frame, load_raw          # noqa: E402
from config import ID, OUT, TARGET                           # noqa: E402
from features import add_features                            # noqa: E402
from modeling import reason_codes                            # noqa: E402
from prioritise import TIER_ACTION, apply_tiers, play, score  # noqa: E402


def predict_file(input_path, model_path=OUT / "retainiq_model.pkl", cutoffs_path=OUT / "tier_cutoffs.json"):
    bundle = joblib.load(model_path)
    cut = json.load(open(cutoffs_path))
    log = CleaningLog()
    raw, log = load_raw(input_path, log)
    df = clean_frame(raw, log, verbose=False)
    true_label = df[TARGET].copy() if TARGET in df else None
    df[TARGET] = np.nan                                  # never used for scoring
    feats = bundle["features"]
    df = add_features(df, bundle["feature_params"])
    X = df[feats]
    df["churn_probability"] = bundle["calibrated"].predict_proba(X)[:, 1]
    df["top_reasons"] = reason_codes(bundle["base"], X, df, feats)
    df["num_products"] = df["num_products"].fillna(1)
    df["avg_monthly_balance"] = df["avg_monthly_balance"].fillna(0)
    df = apply_tiers(df, cut)
    out = score(df)
    out["recommended_action"] = out["priority_tier"].map(TIER_ACTION) + " | " + out.apply(play, axis=1)
    cols = [ID, "churn_probability", "risk_tier", "priority_tier", "balance_at_risk", "net_retention_value",
            "priority_score", "priority_rank", "top_reasons", "recommended_action"]
    return out[cols], log.frame(), (true_label.set_axis(df[ID]) if true_label is not None else None)


def main():
    ap = argparse.ArgumentParser(description="RetainIQ: score customers for churn risk and value at risk")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(OUT / "scored_customers.csv"))
    ap.add_argument("--model", default=str(OUT / "retainiq_model.pkl"))
    a = ap.parse_args()
    scored, log, _ = predict_file(a.input, a.model)
    scored.to_csv(a.output, index=False)
    log.to_csv(Path(a.output).with_name(Path(a.output).stem + "_cleaning_log.csv"), index=False)
    print(f"Scored {len(scored):,} customers -> {a.output}")
    print(log[log.rows_affected > 0][["issue", "rows_affected", "action"]].to_string(index=False))
    print(scored.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
