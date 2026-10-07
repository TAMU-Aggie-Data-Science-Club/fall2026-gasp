# Deliverables & Timeline

> **How to read this file.** This is the PMs' best current estimate of what G.A.S.P. needs to ship and roughly when. It is a **living plan, not a contract**. The authoritative picture lives in **GitHub Issues and the Project board**.

## Milestones (suggested)

| # | Deliverable | Description | Owner (role) | Target |
|---|-------------|-------------|--------------|--------|
| 1 | Scoping & onboarding | Everyone pulls EIA weekly storage and plots it against the 5-year range. Define the target (weekly net change in Lower 48 working gas, Bcf), the success metric (MAE in Bcf, plus interval coverage), and the as-of rule: features come only from the reporting week and data published before release. | PM + all | Week 1 |
| 2 | Data ingestion | Reproducible fetchers for EIA weekly storage and NOAA weekly degree days into SQLite, with a `retrieved_at` timestamp on every row. Confirm ~10 years of weekly degree-day history is obtainable. Documented in [`DATA.md`](DATA.md). | Members | Weeks 1–2 |
| 3 | EDA | Injection/withdrawal seasonality, storage vs. 5-year range, weekly change vs. heating and cooling degree days, extreme-weather weeks. | Members | Weeks 2–3 |
| 4 | Feature engineering | Degree days aligned to the EIA reporting week (ending Friday), deviation from normal, lagged storage changes, week-of-year, inventory vs. 5-year average. | Members | Weeks 3–4 |
| 5 | Baseline forecasts | Naive last-value, 5-year average for the same week, SARIMA, SARIMAX with degree days. Walk-forward validation over ~10 years of releases. | Members | Week 4 |
| 6 | ML forecasts | LightGBM / XGBoost with quantile loss for intervals. Compare against baselines on the same walk-forward splits. Feature importance. | Members | Weeks 5–6 |
| 7 | Dashboard | Streamlit: next-report forecast with uncertainty band, model comparison, historical accuracy, storage vs. 5-year range. | Members + PM | Weeks 6–7 |
| 8 | Handoff & retro | Reproducibility check from a fresh clone, short writeup with the results table, lessons learned. | PM | Week 8 |

## Timeline (rough)

```
Week:   1     2     3     4     5     6     7     8
        |-----|-----|-----|-----|-----|-----|-----|-----|
Onboard █████
Ingest  ███████████
EDA           ███████████
Features            ███████████
Baseline                  █████
ML                              ███████████
Dash                                  ███████████
Retro                                             █████
```

## Later (stretch)

Not part of the v1 plan.

- Benchmark against analyst consensus
- Price reaction to storage surprises
- Regional storage forecasts

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the summary.
- **Dates are estimates.** When reality diverges, update the issue and this file if the shift is material.
- **"Done" is defined per issue** via acceptance criteria — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.
