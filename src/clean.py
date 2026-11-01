"""RetainIQ - Stage 1: robust loading, validation and cleaning.

`load_raw()` and `clean_frame()` are shared by training and by predict.py, so a
hidden test file passes through exactly the same rules. Nothing is deleted
silently: every action is appended to a cleaning log.
"""
import json
import re

import numpy as np
import pandas as pd

from config import (BINARY_COLUMNS, EXPECTED_COLUMNS, GENDERS, ID, NUMERIC_COLUMNS,
                    OUT, DATA, RAW_CSV, REGIONS, SEGMENTS, TARGET, VALID_RANGES)

ID_PATTERN = re.compile(r"^CUST\d+$")


class CleaningLog:
    def __init__(self):
        self.rows = []

    def add(self, issue, rule, n, action, rationale, risk):
        self.rows.append(dict(issue=issue, detection_rule=rule, rows_affected=int(n),
                              action=action, rationale=rationale, risk=risk))

    def frame(self):
        return pd.DataFrame(self.rows)


def _to_number(s: pd.Series) -> pd.Series:
    """'4,485' / ' 1,18,776 ' / '12.5%' -> float; anything unparseable -> NaN."""
    s = s.astype(str).str.strip().str.replace(r"[,\s%]", "", regex=True)
    s = s.where(~s.isin(["", "nan", "None", "NA", "-"]))
    return pd.to_numeric(s, errors="coerce")


def load_raw(path, log: CleaningLog = None) -> pd.DataFrame:
    """Read any CSV with the challenge schema, tolerating junk headers,
    pasted summary rows, extra columns, text-formatted numbers and any column order."""
    log = log or CleaningLog()
    # read ragged rows with the csv module: junk title lines / pasted tables may
    # have a different number of fields than the data, which breaks pd.read_csv
    import csv
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    width = max(len(r) for r in rows)
    raw = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=str)
    # 1. auto-detect the header row (first row containing 'customer_id')
    hdr = None
    for i in range(min(len(raw), 50)):
        if raw.iloc[i].astype(str).str.strip().str.lower().eq("customer_id").any():
            hdr = i
            break
    if hdr is None:
        raise ValueError("No 'customer_id' header found in first 50 rows")
    if hdr > 0:
        log.add("Junk rows above header", "rows before first row containing 'customer_id'",
                hdr, "dropped", "title text, not data", "none")
    cols = [str(c).strip().lower() for c in raw.iloc[hdr]]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    # 2. drop empty / unnamed columns and columns outside the data dictionary
    keep = [c for c in df.columns if c in EXPECTED_COLUMNS]
    dropped = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if dropped:
        extra = [c for c in dropped if c]
        log.add("Columns outside data dictionary", "column name not in data dictionary",
                len(dropped), f"dropped {extra + (['<empty>'] if '' in dropped else [])}",
                "'Aging' duplicates tenure_years (rebuilt from tenure); empty trailing column has no data",
                "none - tenure band recomputed")
    aging = df["aging"].copy() if "aging" in df.columns else None
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df[[c for c in dict.fromkeys(keep)]]
    # 3. keep only real customer rows (drops pasted pivot / summary rows)
    ids = df[ID].astype(str).str.strip()
    is_cust = ids.str.match(ID_PATTERN)
    if (~is_cust).sum():
        log.add("Non-customer rows (pasted summary/pivot tables)",
                "customer_id not matching ^CUST[0-9]+$", (~is_cust).sum(), "dropped",
                "they are report totals, not customers; would corrupt numeric columns",
                "none")
    df = df[is_cust].copy()
    df[ID] = ids[is_cust]
    if aging is not None:
        aging = aging[is_cust].astype(str).str.strip()
    # 4. ensure every expected column exists
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
            if c != TARGET:
                log.add(f"Missing column '{c}'", "column absent from file", len(df),
                        "created as all-missing; imputed in model pipeline",
                        "keeps schema stable", "feature contributes no signal")
    # 5. type coercion
    text_numbers = 0
    for c in NUMERIC_COLUMNS:
        before = df[c].astype(str).str.strip()
        text_numbers += before.str.contains(",", regex=False).sum()
        num = _to_number(df[c])
        bad = num.isna() & ~before.isin(["", "nan", "None", "NA"])
        if bad.sum():
            log.add(f"Unparseable value in '{c}'", "text that cannot be read as a number",
                    bad.sum(), "set to missing (imputed later)", "cannot guess intended value",
                    "small loss of information")
        df[c] = num
    if text_numbers:
        log.add("Numbers stored as text with thousand separators", "value contains ','",
                text_numbers, "separators stripped, cast to number",
                "Indian-style '1,18,776' and '4,485' are numbers", "none")
    df = df[EXPECTED_COLUMNS].reset_index(drop=True)
    if aging is not None:
        df["_aging_original"] = aging.values
    return df, log


def _norm_cat(s, allowed):
    lut = {a.lower().replace(" ", ""): a for a in allowed}
    out = s.astype(str).str.strip().str.lower().str.replace(" ", "").map(lut)
    return out.fillna("Unknown")


def tenure_band(t):
    return pd.cut(t, [-np.inf, 1, 3, 7, 15, np.inf], right=False,
                  labels=["<1", "1-3", "3-7", "7-15", "15+"]).astype(str)


def clean_frame(df: pd.DataFrame, log: CleaningLog, verbose=True) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    # --- duplicates ------------------------------------------------------
    exact = df.duplicated(subset=EXPECTED_COLUMNS).sum()
    log.add("Exact duplicate rows", "all columns identical", exact,
            "dropped (keep first)" if exact else "none found",
            "same customer counted twice", "none")
    df = df.drop_duplicates(subset=EXPECTED_COLUMNS)
    same_id = df[ID].duplicated(keep=False)
    log.add("Same customer_id with different values", "duplicated customer_id",
            same_id.sum(), "keep most complete record, then first" if same_id.sum() else "none found",
            "one row per customer is required for ranking", "may keep the older record")
    df["_nnull"] = df.isna().sum(axis=1)
    df = df.sort_values([ID, "_nnull"]).drop_duplicates(ID).drop(columns="_nnull")

    # --- categoricals ----------------------------------------------------
    for c, allowed in (("gender", GENDERS), ("region", REGIONS), ("customer_segment", SEGMENTS)):
        unk = (_norm_cat(df[c], allowed) == "Unknown").sum()
        df[c] = _norm_cat(df[c], allowed)
        log.add(f"Unrecognised/missing '{c}'", f"value not in {allowed}", unk,
                "set to 'Unknown'" if unk else "none found", "explicit category, not a guess",
                "none")

    # --- missingness (tested even though the training file has none) -----
    for c in NUMERIC_COLUMNS:
        if c == TARGET:
            continue
        m = df[c].isna().sum()
        if m:
            df[f"miss_{c}"] = df[c].isna().astype(int)
            log.add(f"Missing '{c}'", "null after parsing", m,
                    "kept as NaN + missing-indicator; median-imputed inside model pipeline",
                    "imputer fitted only on training folds", "imputed values are estimates")
    total_missing = int(df[NUMERIC_COLUMNS].drop(columns=[TARGET]).isna().sum().sum())
    log.add("Missing values (all numeric fields)", "isna() count", total_missing,
            "see per-column rows" if total_missing else "none found - logic kept for hidden test file",
            "", "")

    # --- validity rules --------------------------------------------------
    for c, (lo, hi) in VALID_RANGES.items():
        x = df[c]
        below = (x < lo) if lo is not None else pd.Series(False, index=df.index)
        above = (x > hi) if hi is not None else pd.Series(False, index=df.index)
        n = int((below | above).sum())
        if c in ("credit_score", "digital_engagement_score", "nps_score"):
            df[c] = x.clip(lo, hi)
            action = "clipped to valid range" if n else "none found"
        else:
            df.loc[below | above, c] = np.nan
            action = "set to missing (imputed in pipeline)" if n else "none found"
        log.add(f"Out-of-range '{c}'", f"outside [{lo}, {hi}]", n, action,
                "domain limit", "none")
    for c in BINARY_COLUMNS:
        bad = df[c].notna() & ~df[c].isin([0, 1])
        if c == TARGET:
            # an unlabelled file (e.g. hidden test) has no target - that is fine
            pass
        df.loc[bad, c] = np.nan
        log.add(f"Non-binary '{c}'", "value not in {0,1}", bad.sum(),
                "set to missing" if bad.sum() else "none found", "binary flag", "none")
    for c in ("num_products", "branch_visits_last_year", "complaint_count_last_year",
              "last_transaction_days_ago"):
        frac = df[c].notna() & (df[c] % 1 != 0)
        df.loc[frac, c] = df.loc[frac, c].round()
        if frac.sum():
            log.add(f"Non-integer count '{c}'", "fractional count", frac.sum(), "rounded",
                    "counts are integers", "none")

    # --- logical consistency (flag, do not overwrite) ---------------------
    flags = df[["has_credit_card", "has_loan", "has_investment_product"]].sum(axis=1, min_count=1)
    df["flag_product_mismatch"] = (flags > df["num_products"]).astype(int)
    log.add("Product flags exceed num_products",
            "card+loan+investment > num_products", df["flag_product_mismatch"].sum(),
            "flagged (flag_product_mismatch); values kept",
            "cannot tell which field is wrong; flag lets model and analysts see it",
            "num_products may be understated for these rows")
    df["flag_tenure_gt_adult_years"] = (df["tenure_years"] > df["age"] - 18).astype(int)
    log.add("Tenure longer than adult life", "tenure_years > age - 18",
            df["flag_tenure_gt_adult_years"].sum(),
            "flagged; values kept",
            "could be minor/joint accounts opened before 18, or data error; "
            "tenure is still predictive so not altered",
            "tenure may be overstated for young customers")
    df["flag_tenure_gt_age"] = (df["tenure_years"] > df["age"]).astype(int)
    log.add("Tenure longer than age (impossible)", "tenure_years > age",
            df["flag_tenure_gt_age"].sum(), "flagged; values kept", "physically impossible -> data error",
            "tenure unreliable for these rows")
    log.add("Negative NPS", "nps_score < 0", (df["nps_score"] < 0).sum(),
            "kept (valid on -100..100 scale)", "NPS can legitimately be negative", "none")
    df["flag_income_floor"] = (df["monthly_income"] == df["monthly_income"].min()).astype(int) \
        if df["monthly_income"].notna().any() else 0
    log.add("Income censored at floor", "monthly_income == minimum (1,500 in training file)",
            int(np.sum(df["flag_income_floor"])), "flagged (income_floor)",
            "looks like a reporting floor; true income may be lower", "income understates spread")
    df["flag_credit_cap"] = (df["credit_score"] == 850).astype(int)
    log.add("Credit score at ceiling", "credit_score == 850", df["flag_credit_cap"].sum(),
            "flagged (credit_cap)", "cap hides variation among best-credit customers",
            "credit score less informative at the top")

    # --- Aging vs tenure ---------------------------------------------------
    if "_aging_original" in df.columns:
        rebuilt = tenure_band(df["tenure_years"])
        lut = {"Less than 1": "<1", "1 to 3": "1-3", "3 to 7": "3-7", "7 to 15": "7-15", "> 15": "15+"}
        orig = df["_aging_original"].map(lut)
        mism = (orig != rebuilt) & orig.notna()
        right_closed = pd.cut(df["tenure_years"], [-np.inf, 1, 3, 7, 15, np.inf], right=True,
                              labels=["<1", "1-3", "3-7", "7-15", "15+"]).astype(str)
        mism_rc = ((orig != right_closed) & orig.notna()).sum()
        boundary = df.loc[mism, "tenure_years"].isin([1, 3, 7, 15]).mean() if mism.sum() else 0
        log.add("'Aging' band disagrees with tenure_years",
                "band rebuilt as [0,1),[1,3),[3,7),[7,15),15+ vs original",
                mism.sum(),
                "original 'Aging' dropped; tenure_band recomputed from tenure_years",
                f"{boundary:.0%} of mismatches sit exactly on a boundary (1/3/7/15): source mixed "
                f"inclusive/exclusive edges (right-closed convention gives {mism_rc} mismatches). "
                "tenure_years is the primary field",
                "none")
        df = df.drop(columns="_aging_original")
    df["tenure_band"] = tenure_band(df["tenure_years"])

    if verbose:
        print(f"clean_frame: {n0} -> {len(df)} rows")
    return df.reset_index(drop=True)


def quality_scorecard(raw_rows: pd.DataFrame, clean: pd.DataFrame, raw_line_count: int):
    """Before/after scorecard on completeness, validity, uniqueness, consistency."""
    def validity(d):
        ok = pd.Series(True, index=d.index)
        for c, (lo, hi) in VALID_RANGES.items():
            x = pd.to_numeric(d[c], errors="coerce")
            if lo is not None:
                ok &= ~(x < lo)
            if hi is not None:
                ok &= ~(x > hi)
        return ok.mean()

    before_numeric = raw_rows.copy()
    return pd.DataFrame([
        dict(dimension="Rows that are real customers",
             before=f"{len(raw_rows)}/{raw_line_count} lines", after=f"{len(clean)}"),
        dict(dimension="Completeness (non-null cells, data dictionary fields)",
             before=f"{1 - before_numeric[EXPECTED_COLUMNS].isna().mean().mean():.4f}",
             after=f"{1 - clean[EXPECTED_COLUMNS].isna().mean().mean():.4f}"),
        dict(dimension="Numeric columns stored as text", before="yes (balance, e.g. '4,485')",
             after="no (all float/int)"),
        dict(dimension="Validity (rows within all domain ranges)",
             before=f"{validity(raw_rows):.4f}", after=f"{validity(clean):.4f}"),
        dict(dimension="Uniqueness (unique customer_id / rows)",
             before=f"{raw_rows[ID].nunique() / len(raw_rows):.4f}",
             after=f"{clean[ID].nunique() / len(clean):.4f}"),
        dict(dimension="Consistency (rows with no logic flag)",
             before="not measured",
             after=f"{1 - clean[['flag_product_mismatch', 'flag_tenure_gt_age']].max(axis=1).mean():.4f}"),
        dict(dimension="Redundant/undocumented columns", before="Aging + empty column",
             after="removed; tenure_band rebuilt"),
    ])


def main():
    log = CleaningLog()
    raw_line_count = sum(1 for _ in open(RAW_CSV, encoding="utf-8", errors="replace"))
    df, log = load_raw(RAW_CSV, log)
    clean = clean_frame(df, log)

    # ---- post-clean assertions ---------------------------------------------
    checks = {
        "row count == 10,000": len(clean) == 10_000,
        "no duplicate IDs": clean[ID].is_unique,
        "no text in numeric columns": all(np.issubdtype(clean[c].dtype, np.number) for c in NUMERIC_COLUMNS),
        "no missing values": clean[EXPECTED_COLUMNS].isna().sum().sum() == 0,
        "balance max < 2M (no summary rows)": clean["avg_monthly_balance"].max() < 2_000_000,
        "tenure max <= 82": clean["tenure_years"].max() <= 82,
        "churn in {0,1}": set(clean[TARGET].unique()) <= {0, 1},
    }
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    assert all(checks.values()), "cleaning assertions failed"

    # unsupervised feature parameter (no label used): bottom-quartile engagement
    params = {"low_engagement_cutoff": float(clean["digital_engagement_score"].quantile(0.25)),
              "rm_eligible_balance": float(clean.loc[clean["relationship_manager_assigned"] == 1,
                                                     "avg_monthly_balance"].quantile(0.25)),
              "income_floor": float(clean["monthly_income"].min())}
    json.dump(params, open(OUT / "feature_params.json", "w"), indent=2)

    from features import add_features
    feat = add_features(clean, params)
    feat.to_csv(DATA / "cleaned_bank_data.csv", index=False)
    log.frame().to_csv(OUT / "data_cleaning_log.csv", index=False)
    sc = quality_scorecard(df, clean, raw_line_count)
    sc.to_csv(OUT / "data_quality_scorecard.csv", index=False)
    profile = clean[NUMERIC_COLUMNS].describe().T
    profile["n_unique"] = clean[NUMERIC_COLUMNS].nunique()
    profile.to_csv(OUT / "data_profile.csv")
    print(log.frame()[["issue", "rows_affected", "action"]].to_string())
    print(sc.to_string())
    print(json.dumps(params, indent=2))
    return feat


if __name__ == "__main__":
    main()
