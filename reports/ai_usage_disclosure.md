# AI usage disclosure

## Tools used

**Claude (Anthropic), via Claude Code**, was used as a coding and analysis assistant. It wrote the Python pipeline in `src/`, the figures, the notebooks and the first drafts of these reports, working from the challenge brief and a detailed team specification. No real customer data was used: only the supplied simulated file, processed locally, with no external upload of the data.

## How outputs were verified

1. **Every number is computed, not typed.** All metrics come from scripts run end-to-end on the raw CSV (`python src/run_all.py`, ~100 s, fixed seed 42, pinned `requirements.txt`). `reports/claims_register.md` is generated from the output files.
2. **Independent reproduction of known facts.** The pipeline had to reproduce baseline facts observed in the raw file before analysis began: 10,000 rows, 15.78% churn, 6.6% churned balance, the segment / product / complaint / inactivity churn rates, and the 199 product-flag mismatches. It did.
3. **Automated PASS/FAIL assertions in every stage:**
   - row counts, no text in numeric columns, no duplicate IDs;
   - no ID / target / protected attributes in model features;
   - no train/hold-out overlap; synthetic rows built only from training rows;
   - ranked list contains only active customers;
   - balanced pilot arms;
   - unseen-data demo AUC matches the reported hold-out AUC.
4. **Bugs caught by verification, then fixed:**
   - A junk title line with fewer fields broke the CSV loader. This was found by the dirty-file demo; the loader now reads ragged rows.
   - An edit accidentally commented out two features, which showed up as an unexplained AUC drop. It was traced by an ablation and fixed, and an assertion now guards the feature list.
   - A base-rate stress test silently capped positives, which was caught by reading its output.
   - A first-draft sensitivity figure misread Jaccard similarity as overlap; it was corrected to the share of the list that changed.
5. **Sanity bounds and honesty checks.**
   - The value-optimal threshold (0.06, flagging 65% of customers) was reported, not hidden; it led to a capacity-bound operating flag.
   - The finding that synthetic augmentation does not help was kept.
   - A 1-year value horizon made the RM call unprofitable for most P1 customers. That led to an explicit, sensitivity-tested horizon assumption rather than silently changing costs.
6. **Human review.** The team must be able to explain each choice; see `reports/judge_qa.md`.
