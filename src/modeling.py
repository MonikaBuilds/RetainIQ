"""RetainIQ - shared modelling utilities (model zoo, metrics, value, reason codes)."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import FEE_PER_PRODUCT, NIM, SEED, VALUE_HORIZON_YEARS
from features import PLAIN_ENGLISH

SIMPLICITY = ["logreg", "tree", "random_forest", "lightgbm"]   # simplest first


def make_model(name, class_weight=None, **kw):
    imp = ("impute", SimpleImputer(strategy="median"))
    if name == "dummy":
        return Pipeline([imp, ("clf", DummyClassifier(strategy="prior"))])
    if name == "logreg":
        return Pipeline([imp, ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=kw.get("C", 1.0), max_iter=2000,
                                                    class_weight=class_weight))])
    if name == "tree":
        return Pipeline([imp, ("clf", DecisionTreeClassifier(max_depth=kw.get("max_depth", 4), min_samples_leaf=100,
                                                             class_weight=class_weight, random_state=SEED))])
    if name == "random_forest":
        return Pipeline([imp, ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=kw.get("min_samples_leaf", 20),
                                                             max_features="sqrt", class_weight=class_weight,
                                                             n_jobs=-1, random_state=SEED))])
    if name == "lightgbm":
        return Pipeline([imp, ("clf", LGBMClassifier(n_estimators=kw.get("n_estimators", 300), learning_rate=0.03,
                                                     num_leaves=kw.get("num_leaves", 15), min_child_samples=50,
                                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                     reg_lambda=1.0, class_weight=class_weight,
                                                     random_state=SEED, verbose=-1))])
    raise ValueError(name)


GRID = {
    "logreg": [dict(C=0.1), dict(C=1.0)],
    "tree": [dict(max_depth=3), dict(max_depth=5)],
    "random_forest": [dict(min_samples_leaf=10), dict(min_samples_leaf=30)],
    "lightgbm": [dict(num_leaves=7, n_estimators=400), dict(num_leaves=15, n_estimators=300)],
}


def customer_value(df):
    """[ASSUMPTION] value of keeping a customer = (balance x NIM + products x fee) x value horizon (years)."""
    annual = df["avg_monthly_balance"].fillna(0) * NIM + df["num_products"].fillna(1) * FEE_PER_PRODUCT
    return annual * VALUE_HORIZON_YEARS


def topk_metrics(y, p, bal, frac):
    y, p, bal = np.asarray(y), np.asarray(p), np.asarray(bal)
    k = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-p, kind="stable")[:k]
    recall = y[idx].sum() / max(1, y.sum())
    precision = y[idx].mean()
    lift = precision / y.mean()
    bal_capture = (bal[idx] * y[idx]).sum() / max(1e-9, (bal * y).sum())
    return recall, precision, lift, bal_capture


def all_metrics(y, p, bal):
    y, p = np.asarray(y), np.asarray(p)
    out = dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
               brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    for f in (0.05, 0.10, 0.20):
        r, pr, l, b = topk_metrics(y, p, bal, f)
        tag = int(f * 100)
        out[f"recall_top{tag}"], out[f"precision_top{tag}"] = r, pr
        out[f"lift_top{tag}"], out[f"balance_capture_top{tag}"] = l, b
    return out


def bootstrap_ci(y, p, bal, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p, bal = np.asarray(y), np.asarray(p), np.asarray(bal)
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        rows.append(all_metrics(y[i], p[i], bal[i]))
    b = pd.DataFrame(rows)
    point = all_metrics(y, p, bal)
    return {k: dict(value=float(point[k]), ci_low=float(b[k].quantile(0.025)), ci_high=float(b[k].quantile(0.975)))
            for k in point}


def psi(a, b, bins=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    pa = np.histogram(a, qs)[0] / len(a) + 1e-6
    pb = np.histogram(b, qs)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


# ------------------------------------------------------------ reason codes
def _describe(feat, raw):
    v = raw.get(feat, np.nan)
    if feat in ("last_transaction_days_ago", "log_days_inactive"):
        return f"{int(raw['last_transaction_days_ago'])} days since last transaction"
    if feat == "inactive_45plus":
        return f"dormant ({int(raw['last_transaction_days_ago'])} days, no recent activity)"
    if feat in ("num_products", "single_product", "product_flag_count", "products_per_tenure"):
        n = int(raw["num_products"])
        return "single product" if n == 1 else f"only {n} products"
    if feat == "complaints_grp":
        return f"{int(raw['complaint_count_last_year'])} complaint(s) last year"
    if feat in ("digital_engagement_score", "low_engagement"):
        return f"low digital engagement ({raw['digital_engagement_score']:.0f}/100)"
    if feat in ("log_balance", "balance_to_income"):
        return f"low balance ({raw['avg_monthly_balance']:,.0f})"
    if feat == "segment_rank":
        return f"{raw['customer_segment']} segment"
    if feat == "tenure_years":
        return f"short tenure ({raw['tenure_years']:.1f} yrs)"
    if feat == "relationship_manager_assigned":
        return "no relationship manager"
    if feat == "rm_eligible_unassigned":
        return f"high balance ({raw['avg_monthly_balance']:,.0f}) but no RM"
    if feat in ("flag_product_mismatch", "flag_tenure_gt_adult_years"):
        return PLAIN_ENGLISH[feat]
    if feat == "nps_score":
        return f"low NPS ({int(raw['nps_score'])})"
    if feat == "branch_visits_last_year":
        return f"{int(raw['branch_visits_last_year'])} branch visits"
    if feat == "credit_score":
        return f"credit score {int(raw['credit_score'])}"
    if feat == "log_income":
        return f"income {raw['monthly_income']:,.0f}"
    return PLAIN_ENGLISH.get(feat, feat) + (f" = {v}" if not pd.isna(v) else "")


def contributions(base_pipeline, X):
    """Per-row additive contributions (log-odds scale) for the uncalibrated base model."""
    clf = base_pipeline.named_steps["clf"]
    Xt = base_pipeline[:-1].transform(X)
    if isinstance(clf, LogisticRegression):
        return Xt * clf.coef_[0]
    if isinstance(clf, LGBMClassifier):
        return clf.predict(Xt, pred_contrib=True)[:, :-1]
    import shap
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    return sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)


def reason_codes(base_pipeline, X, raw, features, top=3):
    C = contributions(base_pipeline, X)
    out = []
    raw = raw.reset_index(drop=True)
    for i in range(len(C)):
        order = np.argsort(-C[i])
        seen, parts = set(), []
        for j in order:
            if C[i, j] <= 0 or len(parts) >= top:
                break
            txt = _describe(features[j], raw.iloc[i])
            key = txt.split(" (")[0]
            if key in seen:
                continue
            seen.add(key); parts.append(txt)
        out.append(" + ".join(parts) if parts else "no strong risk signal")
    return out
