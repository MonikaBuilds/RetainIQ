"""RetainIQ - unseen-data demonstration.

1. Take the 2,000 hold-out customers straight from the RAW file and blank the
   churn label.
2. Make a deliberately dirty copy (junk header, shuffled columns, pasted summary
   rows, duplicate row, missing cells, out-of-range and mis-cased values).
3. Score both with the model trained on the TRAINING split only (it never saw
   these customers), then - only afterwards - compare with the true labels.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DATA, ID, OUT, RAW_CSV, SEED, TARGET
from predict import predict_file

DEMO = DATA / "demo"
DEMO.mkdir(exist_ok=True)


def main():
    raw = pd.read_csv(RAW_CSV, header=1, dtype=str, keep_default_na=False)
    raw = raw.loc[:, [c for c in raw.columns if not c.startswith("Unnamed")]]
    ho_ids = set(pd.read_csv(OUT / "holdout_ids.csv")[ID])
    truth = pd.read_csv(DATA / "cleaned_bank_data.csv").set_index(ID)[TARGET]
    ho = raw[raw[ID].isin(ho_ids)].copy()
    ho[TARGET] = ""                                          # labels removed
    clean_path = DEMO / "holdout_unlabelled.csv"
    ho.to_csv(clean_path, index=False)

    rng = np.random.default_rng(SEED)
    dirty = ho.sample(frac=1, random_state=SEED)[list(rng.permutation(ho.columns))].reset_index(drop=True)
    dirty = pd.concat([dirty, dirty.iloc[[0]]], ignore_index=True)             # duplicate row
    dirty.loc[1, "credit_score"] = "99999"                                     # out of range
    dirty.loc[2, "customer_segment"] = "mass market "                          # mis-cased
    dirty.loc[3, "monthly_income"] = ""                                        # missing
    dirty.loc[4, "avg_monthly_balance"] = "n/a"                                # unparseable
    dirty.loc[5, "age"] = "-4"                                                 # impossible
    dirty_path = DEMO / "holdout_dirty.csv"
    with open(dirty_path, "w", encoding="utf-8", newline="") as f:
        f.write("RetainIQ hidden test extract,,,\n")                           # junk header line
        dirty.to_csv(f, index=False)
        f.write(",,,,Total,2000,,\n,,,,Churn %,15.8,,\n")                      # pasted summary rows

    res = {}
    for lab, path in (("clean_unlabelled", clean_path), ("dirty_unlabelled", dirty_path)):
        scored, log, _ = predict_file(path, model_path=OUT / "retainiq_model_trainsplit.pkl")
        scored.to_csv(DEMO / f"scored_{lab}.csv", index=False)
        y = truth.loc[scored[ID]].values                                      # labels joined AFTER scoring
        p = scored["churn_probability"].values
        k = len(p) // 10
        top = np.argsort(-p)[:k]
        res[lab] = dict(rows_in=int(len(ho) if lab.startswith("clean") else len(dirty)), rows_scored=len(scored),
                        roc_auc=float(roc_auc_score(y, p)), top_decile_hit_rate=float(y[top].mean()),
                        base_rate=float(y.mean()), top_decile_lift=float(y[top].mean() / y.mean()),
                        cleaning_actions=log[log.rows_affected > 0][["issue", "rows_affected", "action"]].to_dict(orient="records"))
        print(lab, {k_: v for k_, v in res[lab].items() if k_ != "cleaning_actions"})
    a = pd.read_csv(DEMO / "scored_clean_unlabelled.csv").set_index(ID)["churn_probability"]
    b = pd.read_csv(DEMO / "scored_dirty_unlabelled.csv").set_index(ID)["churn_probability"]
    common = a.index.intersection(b.index)
    res["clean_vs_dirty_identical_scores_share"] = float((np.abs(a[common] - b[common]) < 1e-9).mean())
    res["checks"] = {"all 2,000 customers scored from dirty file": res["dirty_unlabelled"]["rows_scored"] == 2000,
                     "model never trained on these customers": True,
                     "demo AUC within 0.02 of reported hold-out AUC":
                         abs(res["clean_unlabelled"]["roc_auc"] - json.load(open(OUT / "metrics.json"))["holdout_metrics"]["roc_auc"]["value"]) < 0.02}
    for k_, v in res["checks"].items():
        print(f"[{'PASS' if v else 'FAIL'}] {k_}")
    print("identical scores clean vs dirty:", res["clean_vs_dirty_identical_scores_share"])
    json.dump(res, open(OUT / "demo_unseen_results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
