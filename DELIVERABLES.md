# Deliverables & Timeline

> **How to read this file.** This is the PMs' best current estimate of what AgriCast needs to ship and roughly when. It is a **living plan, not a contract**. The authoritative picture lives in **GitHub Issues and the Project board**.

## Milestones (suggested)

| # | Deliverable | Description | Owner (role) | Target |
|---|-------------|-------------|--------------|--------|
| 1 | Project scoping | Pick commodity + horizon. Define success metric (MAE, MAPE, calibration of intervals). Define what a "point-in-time" feature set means. | PM | Week 1 |
| 2 | Data ingestion | Reproducible fetchers for USDA NASS + WASDE + CME futures. Cache raw snapshots. Documented in [`DATA.md`](DATA.md). | Members | Weeks 1–2 |
| 3 | EDA | Long-run price series, WASDE surprise reactions, planting progress vs. yield relationships. | Members | Week 2 |
| 4 | Feature engineering | Fundamentals (stocks, exports, planting) + market features (lagged returns, roll-adjusted futures) with proper as-of alignment. | Members | Weeks 3–4 |
| 5 | Baseline forecasts | Naive last-value, SARIMAX, Prophet. Walk-forward CV. | Members | Week 4 |
| 6 | ML forecasts | XGBoost / LightGBM with quantile loss for intervals. Compare against baselines on the same walk-forward splits. | Members | Weeks 5–6 |
| 7 | Dashboard | Forecast + uncertainty band, fundamentals view, feature-attribution panel. | Members + PM | Weeks 6–7 |
| 8 | Handoff & retro | Reproducibility check, short writeup, lessons learned. | PM | Week 8 |

## Timeline (rough)

```
Week:   1     2     3     4     5     6     7     8
        |-----|-----|-----|-----|-----|-----|-----|
Scope   ██
Ingest        ████
EDA                 ██
Features                  ████
Baseline                        ██
ML models                            ████
Dashboard                                  ████
Retro                                              ██
```

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the summary.
- **Dates are estimates.** When reality diverges, update the issue and this file if the shift is material.
- **"Done" is defined per issue** via acceptance criteria — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.
