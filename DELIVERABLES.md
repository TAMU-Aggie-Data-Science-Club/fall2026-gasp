# Deliverables & Timeline

> **How to read this file.** This is the PMs' best current estimate of what G.A.S.P. needs to ship and roughly when. It is a **living plan, not a contract**. The authoritative picture lives in **GitHub Issues and the Project board**.

## Milestones (suggested)

| # | Deliverable | Description | Owner (role) | Target |
|---|-------------|-------------|--------------|--------|
| 1 | Scoping & onboarding | Everyone pulls EIA weekly storage and plots it against the 5-year range. Define the target (weekly net change in Lower 48 working gas, Bcf), the success metric (MAE in Bcf, plus interval coverage), and the as-of rule: features come only from the reporting week and data published before release. | PM + all | Week 1 |
| 2 | Core modules | Team A: EIA API client, SQLite storage with `retrieved_at` point-in-time reads, data validation checks. Team B: CPC weekly degree-day parser, degree-day math and Friday-week alignment from daily temperatures; confirm ~10 years of weekly degree-day history is obtainable. Team C: storage features (calendar, lags, 5-year stats with no leakage), naive baselines, walk-forward backtest with MAE, RMSE, and bias. Shapes passed between modules are defined in [`CONTRACTS.md`](CONTRACTS.md). | Teams A, B, C | Week 2 |
| 3 | Pipeline & weather integration | `pipeline.py` wires the Team A modules into one fetch, validate, store run, with a run log. Degree days from the chosen source (CPC or NOAA daily) loaded into SQLite. Degree-day features merged onto storage features: aligned to the EIA reporting week (ending Friday), deviation from normal. EDA: weekly change vs. heating and cooling degree days. First results table: naive baselines through the backtest on pipeline data. | Teams A, B, C | Week 3 |
| 4 | Time-series forecasts | SARIMA and SARIMAX with degree days, built as forecasters that plug into the backtest. Added to the results table alongside the naive baselines. | Members | Week 4 |
| 5 | ML forecasts | LightGBM / XGBoost with quantile loss for intervals. Interval coverage added to the backtest. Compared against all earlier models on the same walk-forward splits. Feature importance. | Members | Weeks 5–6 |
| 6 | Dashboard | Streamlit: next-report forecast with uncertainty band, model comparison, historical accuracy, storage vs. 5-year range. | Members + PM | Weeks 6–7 |
| 7 | Handoff & retro | Reproducibility check from a fresh clone, short writeup with the results table, lessons learned. | PM | Week 8 |

## Timeline (rough)

```
Week:   1     2     3     4     5     6     7     8
        |-----|-----|-----|-----|-----|-----|-----|-----|
Onboard █████
Modules       █████
Pipeline            █████
SARIMA                    █████
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
