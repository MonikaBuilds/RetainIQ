# RetainIQ: judge Q&A (defended answers)

Numbers trace to `reports/claims_register.md`.

**1. Target and horizon?**
`churned` = 1 if the customer left. The file has no dates, so there is no horizon. The model finds customers who *resemble past leavers*, and probabilities are calibrated to this snapshot's 15.8% base rate. We say this rather than invent a "90-day" claim.

**2. Did you keep `last_transaction_days_ago` and NPS? What happens without them?**
We ran a pre-registered three-way test on the hold-out: AUC 0.845 with everything, 0.821 without days-inactive, 0.812 without both. Inactivity may be a *symptom* of leaving, so we deploy B (without it). That loses 0.025 AUC but warns *before* the customer goes quiet. Inactivity stays as a live "Silent Exits" trigger: 46+ days inactive goes to a re-engagement journey.

**3. Does an RM reduce churn, or just mark rich customers?**
Mostly it marks them. RMs cover 81% of Affluent and Private customers but 6% of Mass Market. Adjusting for segment, balance, tenure and products moves the odds ratio from 0.24 to 0.38 [0.30–0.50]. Some association remains in every segment, but this is observational, so we make no causal claim. "Assign an RM to high-balance unassigned customers" is a *test* in the pilot, not a proven lever.

**4. Your 50k data is synthetic. Why trust any metric?**
No reported metric uses synthetic rows. Every metric comes from the 2,000 real hold-out customers, which were never augmented. The synthetic rows were built only from training-split customers (lineage-tagged and asserted) and pass validation: discriminator AUC 0.55, churn rate identical, maximum KS 0.06. They did *not* improve the model (0.821 vs 0.819), so we deploy on real data only. A negative result, reported honestly.

**5. Why isn't accuracy the headline?**
Predicting that nobody leaves scores 84.2% accuracy and is useless. We report PR-AUC (0.499 vs 0.158 random), lift (3.45× in the top 10%) and the balance captured.

**6. How was the threshold chosen, and what does a false positive cost?**
By expected money on training out-of-fold predictions. A false positive costs the contact (25 blended; 5 digital to 150 RM). A false negative costs the forgone saveable value: P × value × save rate. With cheap contacts, almost any threshold pays (the value-optimal t = 0.06 would flag 65% of customers). The real constraint is team capacity, so the operating flag is the top 25% of risk (t = 0.22), which is also the tier cut.

**7. Are the probabilities calibrated?**
Yes. On the hold-out the Brier score is 0.104, and the mean prediction is 16.3% against 15.8% actual. The reliability curve tracks the diagonal (Figure 11). Platt scaling was chosen by cross-validated Brier on training data. It is calibrated within every segment (±0.01). Under a base-rate shift the ranking holds but probabilities drift, so we recalibrate monthly.

**8. Why rank by probability × balance instead of probability alone?**
In the back-test with actual outcomes and 500 contacts, the combined rule catches 36% of the churned balance against 9% for risk alone. It catches fewer churners (96 vs 317), and that is the point: the bank's loss is deposits, not headcount.

**9. What stops one millionaire dominating the list?**
The largest single customer is 3.2% of the top-100's balance at risk. Winsorising balances at the 99th percentile changes only 2% of the top 250. The top-ranked customer has a 1.1m balance but only 8.5% risk, so they are routed to P2 (cheap nurture), not a discount.

**10. Does the model treat gender, age or region differently?**
None of the three is a model input. AUC differs by 0.018 between genders and the tier mix is near-identical (P1: 1.5% of women vs 1.3% of men). Churn itself is flat by region and gender (Cramer's V 0.02 / 0.001). Possible proxies, such as tenure for age, are disclosed in the model card.

**11. What if the margin assumption is wrong by half?**
Halving the margin changes 17% of the top-250. A 1-year instead of 3-year horizon changes 23%, which makes it our most sensitive assumption. P1 membership doesn't depend on margin at all; it's defined by risk and balance quartiles. The pilot measures the real save rate and value.

**12. What about 5 complaints with n = 1?**
We never quote it. Complaints are grouped as 0 / 1 / 2 / 3+ (3+ has n = 67, with a wide CI). The trend test uses grouped counts.

**13. How will you know the intervention caused the retention?**
Randomisation. Within the same high-risk pool, stratified by tier × segment × risk decile, half get treatment and half get nothing. The difference in churn is the causal effect (intention-to-treat). The model's selection is identical in both arms.

**14. Different column order, or text in a numeric column?**
Tested. The demo file had a junk title line, shuffled columns, pasted summary rows, a duplicate row, "n/a", a blank cell, credit score 99999 and age −4. All 2,000 customers were scored, and 99.85% of scores are identical to the clean file. Every action is logged.

**15. Three charts for the CEO?**
- Figure 3: churners vs where the money is (top quintile = 7% of churners, 56% of churned balance).
- Figure 17: our rule catches 4× the leaving money of risk-only ranking.
- Figure 18: the WorthSaving matrix, i.e. who gets what.

**16. Single biggest limitation?**
No time stamps and no treatment history. We can rank customers who look like past leavers, but we can't yet say who is *persuadable* or how fast they leave. The pilot fixes the first; dated data would fix the second.

**17. How would you detect the model going stale?**
Monthly AUC on realised outcomes, PSI per feature (now 0.006) and calibration-in-the-large. Retrain if AUC < 0.78, PSI > 0.2, or the predicted vs observed gap exceeds 3 points.

**18. Why a simpler model over the best scorer?**
Logistic regression was the best *or* within one standard error in every variant. LightGBM and random forest added nothing measurable, while logistic regression gives exact reason codes and stable drivers (the top four drivers were the same in 5 of 5 seeds).

**19. How did you verify AI-generated code?**
See `ai_usage_disclosure.md`:
- reproduction of known facts;
- PASS/FAIL assertions;
- an unseen-data demo matching the hold-out AUC;
- a one-command rerun;
- four real bugs found by these checks and fixed.

**20. Only 200 customers: who and what?**
The top 200 by net retention value (`outputs/ranked_customers.csv`): 30 P1, 138 P2 and 32 P3, mostly Private Banking and Affluent. They hold 26% of all active balance at risk. Each gets their tier action plus driver-specific plays; complaint recovery and RM/concierge assignment dominate.

**21. Why "RetainIQ" and "WorthSaving"?**
Because the decision isn't "who will leave?" but "who is worth spending to keep?". The name carries the balance-weighted logic, not just the model.

**22. The 30-second pitch.**
"Churners are 16% of customers but hold under 7% of deposits, so chasing every leaver wastes money. RetainIQ scores every active customer on how likely they are to leave and how much they'd take, then tells you who to call, who to nudge digitally and who to leave alone. On past data, it would have caught four times more leaving money than a risk-only list. We'll prove the real effect with a randomised pilot on 2,106 customers."
