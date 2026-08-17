# AgriCast — Beginner

*ADSC Catalyst Project · Fall 2026*

## Overview

AgriCast forecasts future prices of major US grain commodities (corn, wheat, soybeans) by combining historical prices with crop production, exports, storage, and planting statistics. The final artifact is a forecasting dashboard that puts a probabilistic price outlook in front of a non-quant decision maker.

## Objective

Predict commodity prices at a chosen horizon (e.g., one month ahead), with calibrated uncertainty bands, and expose the forecasts through a dashboard that also surfaces the fundamentals driving them.

## Suggested tech stack

- **Data processing:** Python, Pandas
- **Modeling / ML:** XGBoost, LightGBM, classical time-series (SARIMAX, Prophet) as baselines
- **Visualization:** Plotly, Matplotlib
- **Data sources:** USDA (NASS Quick Stats, WASDE), CME Group futures

See [`DATA.md`](DATA.md) for concrete data sources and how to access them.

## What team members will gain

- Practical experience with commodity market analytics
- A price-forecasting dashboard grounded in real fundamentals
- Skills that sit at the intersection of economics and data science

## Suggested scope (v1)

Pick **one commodity** (corn is the cleanest) and **one horizon** (one month ahead).

Build:

1. Ingestion for USDA NASS + WASDE + CME futures history,
2. Feature engineering (planting progress, ending stocks, export sales, lagged futures),
3. Baselines: naive last-value, SARIMAX; then XGBoost / LightGBM,
4. Proper time-series validation (walk-forward, no leakage),
5. A forecast + uncertainty band + fundamentals-view dashboard.

**Out of scope for v1:** multi-commodity portfolios, weather-driven yield modeling from raster data, options-implied volatility.

See [`DELIVERABLES.md`](DELIVERABLES.md) for the suggested deliverable breakdown and rough timeline.

## Repository map

| File / folder | Purpose |
|---|---|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **Start here.** How the team runs the project on GitHub — PM vs. member roles, the issue → PR → `main` flow, branching, worktrees, reviews. |
| [`DELIVERABLES.md`](DELIVERABLES.md) | Suggested deliverables and rough timeline. A living plan, not a contract. |
| [`DATA.md`](DATA.md) | Suggested data sources, how to access them, and the source register. |
| [`data/`](data/) | Local working folder for datasets. **Git-ignored** — data is never committed. |
| [`AGENTS.md`](AGENTS.md) | Machine-facing workflow rules for AI coding agents. |

## Notes for PMs

This README, [`DELIVERABLES.md`](DELIVERABLES.md), and [`DATA.md`](DATA.md) are **suggestions**, not commitments. Rewrite them as the team scopes the real project.

## Notes for members

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before touching code. Then pick up an issue from the board.
