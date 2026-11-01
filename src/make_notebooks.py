"""Build and execute the four evidence notebooks (code + executed output).

The notebooks call the same src/ functions, so they cannot drift from the pipeline.
    python src/make_notebooks.py
"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from config import ROOT

NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)
SETUP = ("import sys, json, pandas as pd\nsys.path.insert(0, '../src')\n"
         "from IPython.display import Image, display, Markdown\npd.set_option('display.width', 200)")


def fig(name):
    return f"display(Image('../outputs/figures/{name}'))"


BOOKS = {
    "01_cleaning.ipynb": [
        ("md", "# RetainIQ 01 - Data cleaning\n*See*: what is wrong with the raw file and what we did about it. "
               "Every action is logged; nothing is deleted silently."),
        ("code", SETUP),
        ("code", "import clean\nclean.main()"),
        ("code", "pd.read_csv('../outputs/data_cleaning_log.csv')"),
        ("code", "pd.read_csv('../outputs/data_quality_scorecard.csv')"),
        ("code", "pd.read_csv('../outputs/data_profile.csv', index_col=0).round(2)"),
    ],
    "02_eda_clustering.ipynb": [
        ("md", "# RetainIQ 02 - EDA, hypothesis tests, clustering\nDESCRIPTIVE only - real 10,000 rows, no synthetic data. "
               "Language: *associated with*, never *causes*."),
        ("code", SETUP),
        ("code", "import eda\neda.main()"),
        ("code", "E = json.load(open('../outputs/eda_results.json'))\npd.DataFrame(E['bh_adjusted_p'], index=['BH-adjusted p']).T"),
        ("code", "pd.read_csv('../outputs/univariate_auc.csv')"),
        ("code", "json.dumps(E['hypotheses']['H7_rm_confounding'], indent=1)[:2000]"),
        ("code", "pd.read_csv('../outputs/cluster_profiles.csv')"),
    ] + [("code", fig(f)) for f in ["01_churn_by_segment.png", "02_complaints_dose_response.png",
                                    "03_balance_quintile_rate_vs_value.png", "04_products_vs_churn.png",
                                    "05_inactivity_silent_exits.png", "06_tenure_curve.png",
                                    "07_rm_confounding_by_segment.png", "08_heatmap_segment_x_products.png",
                                    "09_cluster_personas.png", "10_correlation_matrix.png"]],
    "03_model.ipynb": [
        ("md", "# RetainIQ 03 - Predict\nPREDICTIVE. Metrics come only from the untouched 20% real hold-out."),
        ("code", SETUP),
        ("code", "import train\ntrain.main()"),
        ("code", "M = json.load(open('../outputs/metrics.json'))\n"
                 "pd.read_csv('../outputs/cv_model_comparison.csv').round(4)"),
        ("code", "pd.DataFrame({k: {m: v[m] for m in ('roc_auc','pr_auc','recall_top10','balance_capture_top10')} "
                 "for k, v in M['leakage_experiment_holdout'].items()}).round(3)"),
        ("code", "pd.DataFrame(M['holdout_metrics']).T.round(3)"),
        ("code", "pd.read_csv('../outputs/imbalance_comparison.csv').round(4)"),
        ("code", "pd.read_csv('../outputs/fairness_audit.csv').round(3)"),
        ("code", "M['stress_tests'], M['psi_max'], M['augmentation']['validation']['real_vs_synthetic_auc']"),
    ] + [("code", fig(f)) for f in ["11_calibration.png", "12_threshold_cost_curve.png", "13_gain_curve.png",
                                    "14_permutation_importance.png", "15_shap_summary.png", "16_partial_dependence.png"]] + [
        ("md", "## Unseen-data demonstration (train-split model, labels removed, dirty file)"),
        ("code", "import demo_unseen\ndemo_unseen.main()"),
    ],
    "04_prioritisation.ipynb": [
        ("md", "# RetainIQ 04 - Prioritise and Act\nRECOMMENDATION. Only active customers are ranked, using out-of-fold probabilities."),
        ("code", SETUP),
        ("code", "import prioritise, pilot_power\nprioritise.main()\npilot_power.main()"),
        ("code", "pd.read_csv('../outputs/tier_summary.csv').round(2)"),
        ("code", "pd.read_csv('../outputs/ranked_customers.csv').head(20)"),
        ("code", "pd.read_csv('../outputs/prioritisation_sensitivity.csv').round(3)"),
        ("code", "pd.read_csv('../outputs/pilot_sizing.csv').round(4)"),
        ("code", fig("17_budget_gain_curves.png")),
        ("code", fig("18_worthsaving_matrix.png")),
    ],
}


def main():
    for name, cells in BOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(c) if t == "md" else nbf.v4.new_code_cell(c) for t, c in cells]
        ExecutePreprocessor(timeout=900, kernel_name="retainiq").preprocess(nb, {"metadata": {"path": str(NB)}})
        nbf.write(nb, NB / name)
        print("executed", name)


if __name__ == "__main__":
    main()
