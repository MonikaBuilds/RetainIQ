# RetainIQ_Dashboard: UX research brief

## 1. Who uses it, and for what decision

| User | Their question | What they need to do | What they must never misread |
|---|---|---|---|
| **Management committee** (exec) | "Is churn a problem worth funding, and is this credible?" | Grasp the thesis in 30 seconds; trust the numbers | Correlation as cause (RM effect); accuracy as quality |
| **Retention / CRM manager** | "Whom do we contact this month, with what, within budget?" | Set a budget N, get a ranked list, export it | A low-balance high-risk customer as "worth" an RM call |
| **Relationship manager** | "Why is *this* customer on my list and what do I say?" | Open one customer, read the reasons and the play | A probability as a certainty |
| **Analyst / judge** | "Is the model sound? Does it work on new data?" | Inspect validation and leakage; upload a file and see it scored | Synthetic or in-sample numbers as evidence |

## 2. Core tasks (the priority order sets the tab order)

1. Understand the thesis: churners are 15.8% of customers but hold 6.6% of the money.
2. See what goes with churn: who leaves and what signals precede it.
3. Judge whether the model can be trusted.
4. Decide whom to contact for a given budget, and why.
5. Know what to do, and how the pilot will prove it.
6. Score a new file live.

## 3. Information architecture (follows the story arc)

Six tabs:

- **Overview**: four KPI cards and the thesis chart.
- **See**: who leaves, complaints, the money view, Silent Exits, the RM check, and personas.
- **Predict**: scorecard, leakage test, calibration, gain curve, drivers, fairness.
- **Prioritise**: WorthSaving matrix, budget simulator, priority list, customer card.
- **Act**: playbook and pilot.
- **Score a file**: upload a CSV, see it cleaned and scored, download the results.

Two controls apply across tabs:
- **Filters** (sidebar): segment, region, tenure band, RM and persona. They apply to the See and Prioritise tabs.
- **Assumptions** (sidebar): margin, save rate, value horizon and contact cost. They apply to Prioritise, showing live how the list changes when an assumption is wrong.

## 4. Design principles derived from the research

1. **Lead with the money.** Every rate sits next to a count and a balance, because rate alone hides where the value is.
2. **Label the evidence type.** Each tab carries a DESCRIPTIVE, PREDICTIVE or RECOMMENDATION chip, which prevents cause/effect misreadings.
3. **Show uncertainty by default.** Use 95% CI error bars, label n on every bar, suppress cells with n < 30, and group 3+ complaints.
4. **Make assumptions adjustable.** The questions "what if the margin is half?" and "what if the save rate is 10%?" are answered by a slider, not a slide.
5. **One customer, one sentence.** The customer card reads "why + what to do" in plain English before any number.
6. **Separate past from future.** Churned customers appear only in history views; ranking shows active customers only.
7. **Accessible charts.** One colour per meaning (blue = rate/risk, orange = money/priority), no dual axes, direct labels, and the app theme follows the system's light/dark setting.

## 5. Success criteria for the demo

| User | Measure |
|---|---|
| Exec | States the thesis after the Overview tab in under 30 s |
| Manager | Produces an exported top-N list in under 1 minute |
| RM | Explains one customer's risk from the card alone |
| Judge | Uploads the dirty test file and sees 2,000 customers scored, with the cleaning log |
