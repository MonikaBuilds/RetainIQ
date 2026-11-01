"""RetainIQ - Stage 3: PREDICT.

Reported metrics come ONLY from the stratified 20% real hold-out, which is never
augmented, never used for tuning, threshold choice or calibration.
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

import augment
from config import (CV_FOLDS, CV_REPEATS, DATA, DEFAULT_CONTACT_COST, FIG, ID, N_BOOT, OUT,
                    AUG_TARGET_TOTAL, EXPECTED_COLUMNS, SAVE_RATE, SEED, TARGET, HOLDOUT_FRAC)
from features import FEATURE_RATIONALE, VARIANTS, add_features
from modeling import (GRID, SIMPLICITY, all_metrics, bootstrap_ci, customer_value, make_model,
                      psi, reason_codes, topk_metrics)

warnings.filterwarnings("ignore")
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#b8b7b1", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ececea", "axes.titleweight": "bold", "axes.titlesize": 10.5})
M = {}          # everything that goes to metrics.json
CHECKS = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cv_eval(model, X, y, cv):
    r = cross_validate(model, X, y, cv=cv, scoring={"roc": "roc_auc", "pr": "average_precision",
                                                    "brier": "neg_brier_score"}, n_jobs=1)
    return dict(roc_mean=r["test_roc"].mean(), roc_sd=r["test_roc"].std(),
                pr_mean=r["test_pr"].mean(), pr_sd=r["test_pr"].std(),
                pr_se=r["test_pr"].std() / np.sqrt(len(r["test_pr"])), brier=-r["test_brier"].mean())


def rule_score(d):
    """Rule-of-thumb baseline: complaints + inactivity, no model."""
    return d["complaints_grp"].fillna(0) + (d["last_transaction_days_ago"] > 45) + (d["last_transaction_days_ago"] > 90)


def main():
    df = pd.read_csv(DATA / "cleaned_bank_data.csv")
    params = json.load(open(OUT / "feature_params.json"))
    y = df[TARGET].values
    tr_idx, ho_idx = train_test_split(np.arange(len(df)), test_size=HOLDOUT_FRAC, stratify=y, random_state=SEED)
    tr, ho = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)
    ho[[ID]].to_csv(OUT / "holdout_ids.csv", index=False)
    M["split"] = dict(train=len(tr), holdout=len(ho), train_churn=float(tr[TARGET].mean()),
                      holdout_churn=float(ho[TARGET].mean()), seed=SEED, stratified=True)
    CHECKS["no ID overlap train/hold-out"] = len(set(tr[ID]) & set(ho[ID])) == 0
    for v, f in VARIANTS.items():
        CHECKS[f"no id/target/protected attrs in features ({v})"] = not (set(f) & {ID, TARGET, "gender", "age", "region"})
    log(f"split train={len(tr)} holdout={len(ho)}")

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED)

    # ---------------------------------------------------------------- model zoo x leakage variants
    rows = []
    best_by_variant = {}
    for v, feats in VARIANTS.items():
        Xtr = tr[feats]
        for name in SIMPLICITY:
            best = None
            for g in GRID[name]:
                r = cv_eval(make_model(name, **g), Xtr, tr[TARGET], cv)
                r.update(variant=v, model=name, params=json.dumps(g))
                rows.append(r)
                if best is None or r["pr_mean"] > best["pr_mean"]:
                    best = r
            best_by_variant.setdefault(v, {})[name] = best
        r = cv_eval(make_model("dummy"), Xtr, tr[TARGET], cv); r.update(variant=v, model="dummy", params="{}")
        rows.append(r)
        log(f"variant {v} done")
    cvres = pd.DataFrame(rows)
    cvres.to_csv(OUT / "cv_model_comparison.csv", index=False)

    # simplest model within one standard error of the best (per variant)
    chosen = {}
    for v, d in best_by_variant.items():
        top = max(d.values(), key=lambda r: r["pr_mean"])
        for name in SIMPLICITY:
            if d[name]["pr_mean"] >= top["pr_mean"] - top["pr_se"]:
                chosen[v] = (name, json.loads(d[name]["params"]))
                break
        M.setdefault("model_selection", {})[v] = dict(
            best_model=top["model"], best_pr=top["pr_mean"], best_pr_se=top["pr_se"],
            chosen=chosen[v][0], chosen_params=chosen[v][1], chosen_pr=d[chosen[v][0]]["pr_mean"],
            rule="simplest of [logreg, tree, random_forest, lightgbm] within 1 SE of best CV PR-AUC")
    log(f"chosen per variant: {chosen}")

    # ---------------------------------------------------------------- leakage experiment on hold-out
    leak = {}
    rule_ho = rule_score(ho) / 5.0      # rank score scaled to [0,1]; its Brier/log-loss are not meaningful
    leak["rule_of_thumb"] = all_metrics(ho[TARGET], rule_ho + 1e-6 * np.random.default_rng(SEED).random(len(ho)),
                                        ho.avg_monthly_balance)
    for v, feats in VARIANTS.items():
        name, g = chosen[v]
        m = make_model(name, **g).fit(tr[feats], tr[TARGET])
        leak[v] = all_metrics(ho[TARGET], m.predict_proba(ho[feats])[:, 1], ho.avg_monthly_balance)
        leak[v]["model"] = name
    M["leakage_experiment_holdout"] = leak
    dA, dC = leak["A_full"]["roc_auc"], leak["C_early_warning"]["roc_auc"]
    log(f"leakage: A={dA:.3f} B={leak['B_no_inactivity']['roc_auc']:.3f} C={dC:.3f}")

    # Deployment decision (pre-stated rule): if removing the two possibly post-exit
    # fields costs < 0.03 ROC-AUC, deploy the early-warning model C (cannot be
    # 'detecting people already leaving'). Otherwise deploy B if it is within 0.03 of A,
    # else deploy A and label it a late-warning tool.
    if dA - dC < 0.03:
        deploy = "C_early_warning"
    elif dA - leak["B_no_inactivity"]["roc_auc"] < 0.03:
        deploy = "B_no_inactivity"
    else:
        deploy = "A_full"
    M["deployment"] = dict(variant=deploy, rule="deploy C if AUC drop A->C < 0.03; else B if A->B < 0.03; else A",
                           auc_drop_A_to_B=dA - leak["B_no_inactivity"]["roc_auc"], auc_drop_A_to_C=dA - dC)
    feats = VARIANTS[deploy]
    name, g = chosen[deploy]
    log(f"DEPLOY variant={deploy} model={name} {g}")

    # ---------------------------------------------------------------- imbalance handling (CV, train only)
    imb = []
    for nm in ("logreg", "lightgbm"):
        gg = chosen[deploy][1] if nm == name else GRID[nm][0]
        for how in ("none", "class_weight", "smote"):
            if how == "smote":
                base = make_model(nm, **gg)
                mdl = ImbPipeline([("impute", base.named_steps["impute"])] +
                                  ([("scale", base.named_steps["scale"])] if "scale" in base.named_steps else []) +
                                  [("smote", SMOTE(random_state=SEED)), ("clf", base.named_steps["clf"])])
            else:
                mdl = make_model(nm, class_weight="balanced" if how == "class_weight" else None, **gg)
            r = cv_eval(mdl, tr[feats], tr[TARGET], cv); r.update(model=nm, handling=how); imb.append(r)
    M["imbalance_cv"] = imb
    pd.DataFrame(imb).to_csv(OUT / "imbalance_comparison.csv", index=False)
    log("imbalance comparison done")

    # ---------------------------------------------------------------- calibration
    base = make_model(name, **g).fit(tr[feats], tr[TARGET])
    p_raw = base.predict_proba(ho[feats])[:, 1]
    cal = CalibratedClassifierCV(make_model(name, **g), method="isotonic", cv=CV_FOLDS).fit(tr[feats], tr[TARGET])
    p_cal = cal.predict_proba(ho[feats])[:, 1]
    sig = CalibratedClassifierCV(make_model(name, **g), method="sigmoid", cv=CV_FOLDS).fit(tr[feats], tr[TARGET])
    p_sig = sig.predict_proba(ho[feats])[:, 1]
    M["calibration_holdout"] = dict(brier_uncalibrated=float(np.mean((p_raw - ho[TARGET]) ** 2)),
                                    brier_isotonic=float(np.mean((p_cal - ho[TARGET]) ** 2)),
                                    brier_platt=float(np.mean((p_sig - ho[TARGET]) ** 2)),
                                    mean_pred_isotonic=float(p_cal.mean()), actual_rate=float(ho[TARGET].mean()))
    use_sig = M["calibration_holdout"]["brier_platt"] < M["calibration_holdout"]["brier_isotonic"]
    # NOTE the method choice above uses the hold-out only for reporting; the deployed
    # calibrator is chosen by CV Brier on the training data to keep the hold-out clean.
    sk = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    oof_iso = cross_val_predict(CalibratedClassifierCV(make_model(name, **g), method="isotonic", cv=CV_FOLDS),
                                tr[feats], tr[TARGET], cv=sk, method="predict_proba")[:, 1]
    oof_sig = cross_val_predict(CalibratedClassifierCV(make_model(name, **g), method="sigmoid", cv=CV_FOLDS),
                                tr[feats], tr[TARGET], cv=sk, method="predict_proba")[:, 1]
    b_iso, b_sig = np.mean((oof_iso - tr[TARGET]) ** 2), np.mean((oof_sig - tr[TARGET]) ** 2)
    cal_method = "sigmoid" if b_sig < b_iso else "isotonic"
    final_cal, p_ho, oof_tr = (sig, p_sig, oof_sig) if cal_method == "sigmoid" else (cal, p_cal, oof_iso)
    M["calibration_holdout"].update(method_chosen=cal_method, cv_brier_isotonic=float(b_iso), cv_brier_platt=float(b_sig))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for p, lab, col in ((p_raw, "Uncalibrated", GREY), (p_ho, f"Calibrated ({cal_method})", BLUE)):
        fy, fx = calibration_curve(ho[TARGET], p, n_bins=10, strategy="quantile")
        ax.plot(fx, fy, marker="o", color=col, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], ls="--", color=INK, lw=0.8, label="Perfect")
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Observed churn rate (hold-out deciles)")
    ax.legend(frameon=False)
    ax.set_title(f"Calibrated probabilities track reality on the hold-out\n"
                 f"Brier {M['calibration_holdout']['brier_uncalibrated']:.4f} -> "
                 f"{np.mean((p_ho - ho[TARGET]) ** 2):.4f}", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "11_calibration.png", facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------------- headline hold-out metrics + CIs
    ci = bootstrap_ci(ho[TARGET], p_ho, ho.avg_monthly_balance, N_BOOT)
    M["holdout_metrics"] = ci
    M["holdout_baselines"] = dict(dummy=all_metrics(ho[TARGET], np.full(len(ho), tr[TARGET].mean()), ho.avg_monthly_balance),
                                  rule_of_thumb=leak["rule_of_thumb"])
    log(f"HOLDOUT roc={ci['roc_auc']['value']:.3f} [{ci['roc_auc']['ci_low']:.3f},{ci['roc_auc']['ci_high']:.3f}] "
        f"pr={ci['pr_auc']['value']:.3f} lift10={ci['lift_top10']['value']:.2f}")

    # ---------------------------------------------------------------- threshold by expected value (chosen on TRAIN OOF)
    val_tr, val_ho = customer_value(tr).values, customer_value(ho).values
    grid = np.round(np.arange(0.05, 0.91, 0.01), 2)

    def net(yv, p, v, t):
        f = p >= t
        return float((yv[f] * SAVE_RATE * v[f]).sum() - f.sum() * DEFAULT_CONTACT_COST)
    nets_tr = [net(tr[TARGET].values, oof_tr, val_tr, t) for t in grid]
    t_star = float(grid[int(np.argmax(nets_tr))])
    nets_ho = [net(ho[TARGET].values, p_ho, val_ho, t) for t in grid]
    f1s = [f1_score(tr[TARGET], oof_tr >= t) for t in grid]
    t_f1 = float(grid[int(np.argmax(f1s))])
    # Operating flag: with cheap contacts the value-optimal threshold is very low (contacting
    # almost anyone pays), so the binding constraint is team capacity. The operating
    # "high risk" flag is therefore the top quartile of training OOF risk - the same cut
    # that defines the priority tiers.
    t_op = float(np.quantile(oof_tr, 0.75))
    yhat = (p_ho >= t_op).astype(int)
    tn, fp, fn, tp = confusion_matrix(ho[TARGET], yhat).ravel()
    M["threshold"] = dict(operating=t_op, operating_rule="top quartile of training OOF risk (capacity-bound)",
                          value_optimal=t_star, f1_optimal=t_f1, chosen_on="training out-of-fold predictions",
                          value_optimal_flag_share_holdout=float((p_ho >= t_star).mean()),
                          assumptions=dict(save_rate=SAVE_RATE, contact_cost=DEFAULT_CONTACT_COST,
                                           value="(balance x NIM + products x fee) x 3-year horizon"),
                          holdout_at_threshold=dict(precision=float(precision_score(ho[TARGET], yhat)),
                                                    recall=float(recall_score(ho[TARGET], yhat)),
                                                    f1=float(f1_score(ho[TARGET], yhat)), tn=int(tn), fp=int(fp),
                                                    fn=int(fn), tp=int(tp), flagged=int(yhat.sum()),
                                                    net_value=nets_ho[int(np.argmax(nets_tr))],
                                                    accuracy_footnote=float((yhat == ho[TARGET]).mean())),
                          holdout_at_f1_threshold=dict(precision=float(precision_score(ho[TARGET], p_ho >= t_f1)),
                                                       recall=float(recall_score(ho[TARGET], p_ho >= t_f1)),
                                                       f1=float(f1_score(ho[TARGET], p_ho >= t_f1))))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(grid, np.array(nets_tr) / 1000, color=BLUE, lw=2, label="Training (out-of-fold)")
    ax.plot(grid, np.array(nets_ho) * 4 / 1000, color=ORANGE, lw=2, label="Hold-out (scaled x4 to train size)")
    ax.axvline(t_star, color=INK, ls="--", lw=0.8); ax.axhline(0, color=GREY, lw=0.8)
    ax.text(t_star + 0.01, ax.get_ylim()[1] * 0.9, f"value-optimal t = {t_star:.2f}\nflags {(p_ho >= t_star).mean():.0%} of customers", fontsize=8)
    ax.set_xlabel("Probability threshold for contacting"); ax.set_ylabel("Expected net value (thousands)")
    ax.legend(frameon=False, fontsize=8)
    ax.axvline(t_op, color=ORANGE, ls=":", lw=1.2)
    ax.text(t_op + 0.01, ax.get_ylim()[1] * 0.6, f"operating flag t = {t_op:.2f}\n(top 25% risk)", fontsize=8, color=ORANGE)
    ax.set_title("Cheap contacts pay at almost any threshold:\nthe real limit is team capacity, not a cut-off", loc="left")
    ax.set_xlabel(f"Probability threshold  [assumes save rate {SAVE_RATE:.0%}, cost {DEFAULT_CONTACT_COST:.0f}/contact]")
    fig.tight_layout(); fig.savefig(FIG / "12_threshold_cost_curve.png", facecolor="white"); plt.close(fig)

    # gain curve
    order = np.argsort(-p_ho)
    cum_ch = np.cumsum(ho[TARGET].values[order]) / ho[TARGET].sum()
    cum_bal = np.cumsum((ho[TARGET] * ho.avg_monthly_balance).values[order]) / (ho[TARGET] * ho.avg_monthly_balance).sum()
    xs = np.arange(1, len(ho) + 1) / len(ho)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(xs, cum_ch, color=BLUE, lw=2, label="Churners captured")
    ax.plot(xs, cum_bal, color=ORANGE, lw=2, label="Churned balance captured")
    ax.plot([0, 1], [0, 1], ls="--", color=INK, lw=0.8, label="Random")
    ax.set_xlabel("Share of customers contacted (ranked by risk)"); ax.set_ylabel("Cumulative share captured")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Top 20% by risk captures {ci['recall_top20']['value']:.0%} of churners (hold-out)\n"
                 "but less of the churned money - risk alone misses high-balance leavers", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "13_gain_curve.png", facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------------- explainability
    perm = permutation_importance(base, ho[feats], ho[TARGET], scoring="average_precision",
                                  n_repeats=10, random_state=SEED)
    pi = pd.DataFrame({"feature": feats, "importance": perm.importances_mean, "sd": perm.importances_std}) \
        .sort_values("importance", ascending=False)
    pi.to_csv(OUT / "permutation_importance.csv", index=False)
    M["permutation_importance_top8"] = pi.head(8).set_index("feature")["importance"].round(4).to_dict()
    clf = base.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coefs = pd.DataFrame({"feature": feats, "std_coef": clf.coef_[0], "odds_ratio_per_sd": np.exp(clf.coef_[0])}) \
            .sort_values("std_coef", key=np.abs, ascending=False)
        coefs.to_csv(OUT / "logreg_coefficients.csv", index=False)
        M["logreg_top_coefficients"] = coefs.head(10).set_index("feature")["odds_ratio_per_sd"].round(3).to_dict()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    top = pi.head(12).iloc[::-1]
    ax.barh(top.feature, top.importance, xerr=top.sd, color=BLUE, height=0.6, ecolor=INK)
    ax.set_xlabel("Drop in hold-out PR-AUC when the feature is shuffled")
    ax.set_title(f"What the deployed model relies on ({name}, variant {deploy})\n"
                 "correlated wealth fields share credit - read as a group", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "14_permutation_importance.png", facecolor="white"); plt.close(fig)

    # SHAP on a gradient-boosting version for a richer global view
    try:
        import shap
        lgb = make_model("lightgbm", **chosen[deploy][1]) if name == "lightgbm" else make_model("lightgbm", **GRID["lightgbm"][0])
        lgb.fit(tr[feats], tr[TARGET])
        Xs = pd.DataFrame(lgb[:-1].transform(ho[feats]), columns=feats)
        sv = shap.TreeExplainer(lgb.named_steps["clf"]).shap_values(Xs)
        sv = sv[1] if isinstance(sv, list) else sv
        shap.summary_plot(sv, Xs, show=False, max_display=12)
        plt.title("SHAP (LightGBM view): red = high value of the feature", loc="left", fontsize=10)
        plt.tight_layout(); plt.savefig(FIG / "15_shap_summary.png", dpi=150, facecolor="white"); plt.close()
        M["shap_mean_abs_top8"] = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False).head(8).round(4).to_dict()
    except Exception as e:  # pragma: no cover
        M["shap_error"] = str(e)
    pdp_feats = [f for f in ("complaints_grp", "last_transaction_days_ago", "log_balance", "num_products",
                             "digital_engagement_score", "tenure_years") if f in feats][:4]
    fig, ax = plt.subplots(1, len(pdp_feats), figsize=(3.0 * len(pdp_feats), 3.0))
    PartialDependenceDisplay.from_estimator(base, tr[feats], pdp_feats, ax=ax, response_method="predict_proba",
                                            line_kw={"color": BLUE, "lw": 2})
    fig.suptitle("Partial dependence: average predicted churn as one feature varies", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "16_partial_dependence.png", facecolor="white"); plt.close(fig)

    # ---------------------------------------------------------------- stability across seeds
    stab, drivers = [], []
    for s in range(10):
        a, b = train_test_split(np.arange(len(df)), test_size=HOLDOUT_FRAC, stratify=y, random_state=s)
        m = make_model(name, **g).fit(df.iloc[a][feats], y[a])
        p = m.predict_proba(df.iloc[b][feats])[:, 1]
        stab.append(dict(seed=s, roc_auc=roc_auc_score(y[b], p),
                         pr_auc=all_metrics(y[b], p, df.iloc[b].avg_monthly_balance)["pr_auc"]))
        if s < 5:
            pim = permutation_importance(m, df.iloc[b][feats], y[b], scoring="average_precision", n_repeats=5, random_state=s)
            drivers += list(np.array(feats)[np.argsort(-pim.importances_mean)[:5]])
    st = pd.DataFrame(stab)
    M["stability_10_seeds"] = dict(roc_mean=st.roc_auc.mean(), roc_sd=st.roc_auc.std(),
                                   pr_mean=st.pr_auc.mean(), pr_sd=st.pr_auc.std(),
                                   top5_driver_frequency_over_5_seeds=pd.Series(drivers).value_counts().to_dict())
    log(f"stability roc {st.roc_auc.mean():.3f}+-{st.roc_auc.std():.3f}")

    # ---------------------------------------------------------------- fairness audit (hold-out)
    ho_f = ho.copy(); ho_f["p"] = p_ho; ho_f["flag"] = (p_ho >= t_op).astype(int)
    fair = []
    for col in ("customer_segment", "region", "gender", "age_band"):
        for lvl, sub in ho_f.groupby(col):
            pos, neg = sub[sub[TARGET] == 1], sub[sub[TARGET] == 0]
            fair.append(dict(attribute=col, group=lvl, n=len(sub), churn_rate=sub[TARGET].mean(),
                             mean_pred=sub.p.mean(), flag_rate=sub.flag.mean(),
                             auc=roc_auc_score(sub[TARGET], sub.p) if len(pos) >= 10 and len(neg) >= 10 else np.nan,
                             recall=pos.flag.mean() if len(pos) else np.nan,
                             fpr=neg.flag.mean() if len(neg) else np.nan, n_churners=len(pos)))
    fair = pd.DataFrame(fair)
    ov_auc, ov_rec = ci["roc_auc"]["value"], M["threshold"]["holdout_at_threshold"]["recall"]
    fair["auc_gap_flag"] = (ov_auc - fair.auc).abs() > 0.05
    fair["recall_gap_flag"] = ((ov_rec - fair.recall).abs() > 0.10) & (fair.n_churners >= 20)
    fair.to_csv(OUT / "fairness_audit.csv", index=False)
    M["fairness"] = dict(tolerance="AUC gap > 0.05 or recall gap > 0.10 (groups with >=20 churners)",
                         flags=fair[fair.auc_gap_flag | fair.recall_gap_flag][["attribute", "group", "n", "auc", "recall", "fpr"]]
                         .round(3).to_dict(orient="records"),
                         gender_region_ranges={c: dict(auc_range=float(fair[fair.attribute == c].auc.max() - fair[fair.attribute == c].auc.min()),
                                                       recall_range=float(fair[fair.attribute == c].recall.max() - fair[fair.attribute == c].recall.min()))
                                               for c in ("gender", "region", "age_band")})

    # ---------------------------------------------------------------- stress tests
    rng = np.random.default_rng(SEED)
    stress = {}
    def score(X):
        return roc_auc_score(ho[TARGET], final_cal.predict_proba(X)[:, 1])
    stress["baseline"] = score(ho[feats])
    imp = pi.feature.iloc[0]
    X = ho[feats].copy(); X[imp] = np.nan; stress[f"drop_top_feature({imp})"] = score(X)
    X = ho[feats].copy()
    for c in X.columns:
        if X[c].nunique() > 2:
            X[c] = X[c] * rng.normal(1, 0.05, len(X))
    stress["5pct_multiplicative_noise"] = score(X)
    X = ho[feats].copy().astype(float); X = X.mask(rng.random(X.shape) < 0.10); stress["10pct_cells_missing"] = score(X)
    for target_rate in (0.08, 0.30):
        pos, neg = np.where(ho[TARGET] == 1)[0], np.where(ho[TARGET] == 0)[0]
        n_pos = int(target_rate * len(neg) / (1 - target_rate))
        sel = np.r_[neg, rng.choice(pos, n_pos, replace=n_pos > len(pos))]
        pp = final_cal.predict_proba(ho.iloc[sel][feats])[:, 1]
        stress[f"base_rate_{target_rate:.0%}"] = dict(auc=roc_auc_score(ho[TARGET].iloc[sel], pp),
                                                     mean_pred=float(pp.mean()), actual=float(ho[TARGET].iloc[sel].mean()))
    X = ho[feats].copy(); X.loc[X.index[:20], "credit_score"] = 99999; X.loc[X.index[:20], "tenure_years"] = -5
    stress["out_of_range_values_raw_model(no_cleaning)"] = score(X)
    M["stress_tests"] = stress
    log("stress tests done")

    # ---------------------------------------------------------------- drift: PSI train vs hold-out
    M["psi_train_vs_holdout"] = {c: round(psi(tr[c], ho[c]), 4) for c in feats if tr[c].nunique() > 2}
    M["psi_max"] = max(M["psi_train_vs_holdout"].values())

    # ---------------------------------------------------------------- augmentation 10k -> 50k ablation
    n_syn = AUG_TARGET_TOTAL - len(df)
    syn_raw = augment.synthesize(tr[EXPECTED_COLUMNS], n_syn)
    from clean import CleaningLog, clean_frame
    syn_clean = clean_frame(syn_raw[EXPECTED_COLUMNS], CleaningLog(), verbose=False)
    syn = add_features(syn_clean, params)
    syn = syn.merge(syn_raw[[ID, "data_origin", "source_customer_id", "source_neighbor_id"]], on=ID)
    CHECKS["synthetic parents only from training split"] = (set(syn.source_customer_id) | set(syn.source_neighbor_id)) <= set(tr[ID])
    CHECKS["no synthetic rows in hold-out"] = not ho[ID].str.startswith("SYN").any()
    aug = pd.concat([tr.assign(data_origin="real", source_customer_id=tr[ID], source_neighbor_id=""), syn], ignore_index=True)
    aug.to_csv(DATA / "augmented_train.csv", index=False)
    M["augmentation"] = dict(method="class-conditional SMOTE-NC-style interpolation (train split only)",
                             real_train=len(tr), synthetic=len(syn), augmented_train_total=len(aug),
                             grand_total_with_holdout=len(aug) + len(ho),
                             validation=augment.validate(tr, syn))
    abl = {}
    for nm in sorted({name, "lightgbm"}):
        gg = g if nm == name else GRID[nm][0]
        for lab, data in (("real_8k", tr), ("real+synthetic_48k", aug)):
            m = make_model(nm, **gg).fit(data[feats], data[TARGET])
            pp = m.predict_proba(ho[feats])[:, 1]
            r = bootstrap_ci(ho[TARGET], pp, ho.avg_monthly_balance, 300)
            abl[f"{nm}|{lab}"] = {k: r[k] for k in ("roc_auc", "pr_auc", "recall_top10", "brier")}
    M["augmentation"]["ablation_holdout"] = abl
    log("augmentation ablation done")

    # ---------------------------------------------------------------- out-of-fold predictions on all 10k (for ranking)
    oof = cross_val_predict(CalibratedClassifierCV(make_model(name, **g), method=cal_method, cv=CV_FOLDS),
                            df[feats], y, cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    base_full = make_model(name, **g).fit(df[feats], y)
    reasons = reason_codes(base_full, df[feats], df, feats)
    pd.DataFrame({ID: df[ID], "churn_probability_oof": oof, "top_reasons": reasons}).to_csv(OUT / "oof_predictions.csv", index=False)
    M["oof_all_10k"] = dict(roc_auc=roc_auc_score(y, oof), note="cross-fitted: each customer scored by a model that never saw them")

    # model fitted on the TRAINING split only -> used for the honest unseen-data demo on the hold-out
    joblib.dump(dict(calibrated=final_cal, base=base, features=feats, variant=deploy, model_name=name,
                     model_params=g, threshold=t_op, calibration=cal_method, feature_params=params,
                     trained_rows=len(tr), seed=SEED, note="train split only - for hold-out demo"),
                OUT / "retainiq_model_trainsplit.pkl")

    # ---------------------------------------------------------------- final deployable model (fit on all real data)
    final = CalibratedClassifierCV(make_model(name, **g), method=cal_method, cv=CV_FOLDS).fit(df[feats], y)
    joblib.dump(dict(calibrated=final, base=base_full, features=feats, variant=deploy, model_name=name,
                     model_params=g, threshold=t_op, calibration=cal_method, feature_params=params,
                     trained_rows=len(df), seed=SEED), OUT / "retainiq_model.pkl")

    M["feature_table"] = {f: ("include" if f in feats else "include-in-variant") for f in VARIANTS["A_full"]}
    M["feature_rationale"] = FEATURE_RATIONALE
    M["excluded"] = {"customer_id": "identifier", "gender": "protected - audit only", "age": "protected - audit only",
                     "region": "no signal (H9) - audit only", "Aging": "redundant with tenure_years, dropped in cleaning",
                     "churned": "target"}
    for k, v in CHECKS.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    M["checks"] = CHECKS
    json.dump(M, open(OUT / "metrics.json", "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else (bool(o) if isinstance(o, np.bool_) else str(o)))
    log("saved metrics.json and retainiq_model.pkl")
    assert all(CHECKS.values())


if __name__ == "__main__":
    main()

