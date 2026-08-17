# Data

This file explains **AgriCast's** suggested data sources.

> **`DATA.md` is tracked in git. The `data/` folder is not.** Clone the repo, then populate `data/` locally.

## Suggested sources (starting point)

| Source | Origin / URL | Access method | License | Sensitivity | Notes |
|--------|--------------|---------------|---------|-------------|-------|
| USDA NASS Quick Stats | https://quickstats.nass.usda.gov/api | REST API, free key | Public | None | Planting progress, yield, production, stocks by state/commodity |
| USDA WASDE reports | https://www.usda.gov/oce/commodity/wasde | Monthly PDF + XLS | Public | None | Supply/demand balance sheets — the fundamental driver |
| USDA FAS Export Sales | https://apps.fas.usda.gov/esrquery/ | Weekly report, downloadable | Public | None | Weekly export commitments — market-moving |
| CME Group futures | https://www.cmegroup.com/market-data/ | Delayed data public; historical via `yfinance` (ZC=F, ZW=F, ZS=F) | Per-tier | None | Continuous front-month contracts are enough for v1 |
| NOAA weather (US corn belt) | https://www.ncei.noaa.gov/cdo-web/ | REST API, free key | Public | None | Optional — for weather-conditioned features |

## How to think about using each source

- **Release cadence.** USDA WASDE releases monthly on a fixed schedule and is highly market-moving. Any model trained on future WASDE features has leakage. Align features to a "point-in-time" as-of date.
- **Contract rolls.** CME futures contracts expire; use a properly rolled continuous series (`yfinance` handles this reasonably; commercial sources handle it better).
- **Vintage.** USDA revises historical estimates. Cache the *version you trained on* to `data/raw/` — otherwise reproducibility silently breaks.
- **License.** USDA data is public domain. CME live data isn't; delayed / historical for research is fine.

Choosing and vetting a source is a **judgment call** — surface it to a PM rather than deciding a major data direction alone.

## Local layout convention

```
data/
├── raw/          # as fetched — one folder per source, timestamped
├── interim/      # aligned to a common weekly/monthly index, as-of dates preserved
└── processed/    # feature tables and model-ready splits
```

Because `data/` isn't in git, the **pipeline that fetches and builds these folders** is what must be committed and reproducible — not the data itself.
