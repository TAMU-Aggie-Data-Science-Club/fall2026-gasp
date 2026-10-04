# Data

This file explains **G.A.S.P.'s** data sources.

> **`DATA.md` is tracked in git. The `data/` folder is not.** Clone the repo, then populate `data/` locally.

https://www.eia.gov/opendata/browser/natural-gas

## Sources (starting point)

| Source | Origin / URL | Access method | License | Sensitivity | Notes |
|--------|--------------|---------------|---------|-------------|-------|
| EIA Weekly Natural Gas Storage | Natural Gas → Storage → Weekly Working Gas in Underground Storage | REST API v2, free key | Public | None | **The target.** Weekly; returns the storage *level* (Bcf) — the weekly change is derived by differencing. Lower 48 total + 5 regions |
| EIA Natural Gas Production | Natural Gas → Production → Natural Gas Gross Withdrawals and Production | REST API v2 | Public | None | Monthly, ~2-month lag. Filter to dry production, not gross withdrawals |
| EIA Natural Gas Consumption | Natural Gas → Consumption / End Use → Natural Gas Consumption by End Use | REST API v2 | Public | None | Monthly. Residential, commercial, industrial, electric power via the process facet |
| EIA LNG Exports | Natural Gas → Imports and Exports/Pipelines → U.S. Natural Gas Exports and Re-Exports by Country | REST API v2 | Public | None | Monthly. Filter process to LNG. Terminal-level detail available under "by Point of Exit" |
| EIA Mexico Pipeline Exports | Same route as LNG Exports | REST API v2 | Public | None | Monthly. Filter process to pipeline, country to Mexico |
| EIA Canadian Imports | Natural Gas → Imports and Exports/Pipelines → U.S. Natural Gas Imports by Country | REST API v2 | Public | None | Monthly. Filter process to pipeline, country to Canada |
| Henry Hub spot price | Natural Gas → Prices → Natural Gas Spot and Futures Prices (NYMEX) | REST API v2 | Public | None | Context and fuel-switching features; not a forecast target. Futures after April 5, 2024 are not available here — spot only |
| EIA Monthly Supply & Disposition Balance | Natural Gas → Summary → U.S. Natural Gas Monthly Supply and Disposition Balance | REST API v2 | Public | None | Monthly. EIA's own version of the balance — use it to check that our components reconcile |
| NOAA degree days | https://www.ncei.noaa.gov/cdo-web/ | REST API, free key | Public | None | Heating and cooling degree days — the strongest demand driver. CDO is station-level; population weighting is our job |

## How to think about using each source

- **Release timing.** The storage report covers the week ending **Friday** and publishes the following **Thursday** at 10:30 ET. Features must come from the *reporting* week, not the six days between the week's end and publication.
- **Frequency mismatch.** Storage is weekly; consumption and production are monthly with a lag. A method must be built to upsample the monthly data to weekly. Errors are bound to be huge as the dataset is relatively sparse (12 points in a year).
- **Vintage.** EIA revises. Storage gets minor weekly reclassifications; monthly production and consumption are revised for months. Every row stored must carry a `retrieved_at` timestamp so we can reconstruct the data as it stood on any past forecast date.
- **Units.** Storage is Bcf (Billion cubic feet). Production is often Bcf/d (Billion cubic feet per day) or MMcf (Million cubic feet).
- **License.** EIA and NOAA data are public domain.

Choosing and vetting a source is a **judgment call** — surface it to a PM rather than deciding a major data direction alone.

## Getting an EIA API key

https://www.eia.gov/opendata/register.php

Put it in a `.env` file at the repo root. **`.env` is git-ignored — never commit a key.**

```
EIA_API_KEY=your_key_here
```

## Local layout convention

```
data/
├── raw/          # as fetched — one folder per source, timestamped
├── interim/      # aligned to a common weekly/monthly index, as-of dates preserved
└── processed/    # feature tables and model-ready splits
```

Because `data/` isn't in git, the **pipeline that fetches and builds these folders** is what must be committed and reproducible — not the data itself.
