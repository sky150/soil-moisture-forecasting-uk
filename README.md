# Soil Moisture Forecasting (COSMOS-UK)

> Live dashboard: _coming in MS4_ · Video pitch: _coming in MS4_

This live ML system forecasts daily field-scale soil moisture at one agricultural station in the UK's
[COSMOS-UK](https://cosmos.ceh.ac.uk/) network, Balruddery (`BALRD`, Scotland). It ingests soil and weather
observations every day, stores engineered features in a versioned Parquet feature store, and (in later
milestones) retrains a LightGBM model and shows forecasts next to the measurements that arrive afterwards.

**Prediction target:** the daily mean volumetric water content `cosmos_vwc` (%).

**Horizon:** *A forecast is issued on day D using data through D−1; the target is the daily mean soil
moisture of D+1; persistence predicts obs(D−1).* In the code a row is indexed by the target date `d = D+1`,
so every feature uses observations dated `≤ d−2`.

## Architecture

![FTI architecture](docs/architecture.png)

Feature pipeline → feature store (versioned Parquet, GCS) → training pipeline (MS3) → inference + dashboard (MS4).

Why no database, Airflow or Kafka: the data is about one row per day for one station. GitHub Actions already
gives scheduled batch runs next to CI, and object storage with Parquet is enough for a feature store at this size.

## Run it

Requires [uv](https://docs.astral.sh/uv/), a GCS bucket with a service-account key, and, for the image, Docker.

```bash
git clone https://github.com/sky150/soil-moisture-forecasting-uk.git
cd soil-moisture-forecasting-uk
uv sync                      # exact versions from uv.lock
cp .env.example .env         # set STORAGE_URI (gs://...) and GOOGLE_APPLICATION_CREDENTIALS

uv run python -m features backfill             # COSMOS 2023-01-01 .. yesterday (~5 s)
uv run python -m features forecast-backfill    # archived rain forecasts
uv run python -m features forecast-snapshot    # today's live forecast
uv run python -m features build-features       # rebuild the feature table
uv run pytest                                  # tests, no network needed
```

Other commands: `ingest` (re-fetch the last 7 days, the daily job) and `probe` (feed arrival check).
Configuration: the pipeline definition (station, API URLs, parameters, ranges, feature version) is in
[`config/settings.yaml`](config/settings.yaml); deployment values and credentials (`STORAGE_URI`, `GOOGLE_APPLICATION_CREDENTIALS`,
MLflow) are in `.env`. `STORAGE_URI` is required. The tests need neither: they write to a temporary folder.

With Docker:

```bash
docker build -t soil-pipeline .
docker run --rm --env-file .env -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/key.json \
  -v "$GOOGLE_APPLICATION_CREDENTIALS:/secrets/key.json:ro" soil-pipeline ingest
```

Read the features (this is what the training pipeline will do):

```python
import pandas as pd
from common.features import FEATURE_COLUMNS, LABEL
df = pd.read_parquet("gs://<bucket>/soil-moisture/features/v1/BALRD/features.parquet")
train = df[df.is_labelled & (df.index < "2025-01-01")]
X, y = train[FEATURE_COLUMNS], train[LABEL]
```

## Data and features

Sources: COSMOS-UK API (UKCEH, [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)),
collection `1D`; [Open-Meteo](https://open-meteo.com/) Previous Runs API (CC BY 4.0).
Neither API needs a key (Open-Meteo is free for non-commercial use up to 10,000 calls/day; the pipeline makes a handful).

```
<STORAGE_URI>/
  raw/cosmos/<station>/<year>/fetched_at=<ts>.json     API responses, never modified
  raw/openmeteo/<station>/issued_at=<ts>.json          live forecast as issued
  raw/openmeteo/<station>/archive/...                  archived forecasts
  clean/<station>/observations.parquet                 one row per date, newest fetch wins
  clean/<station>/forecasts.parquet                    one row per date, first value wins
  features/v1/<station>/features.parquet               the feature store
  features/latest.json                                 current version + update time
  issued/v1/<station>/target_date=<d>.parquet          inference row frozen on the issue day
  logs/feed_probe.jsonl                                arrival-time probe (temporary)
```

All features are defined once in [`src/common/features.py`](src/common/features.py)
and used by the feature, training and inference pipelines (no training–serving skew). One row per target date `d`;
`L = d−2` is the latest observation available at issue time.

| Column | Definition |
|---|---|
| `vwc_latest` | `obs(L)`; also the persistence forecast |
| `vwc_lag1`, `vwc_lag3`, `vwc_lag7` | `obs(L−1)`, `obs(L−3)`, `obs(L−7)` |
| `vwc_mean3`, `vwc_mean7` | mean soil moisture of the 3 / 7 days ending at `L` |
| `precip_sum3`, `precip_sum7` | rain total of the 3 / 7 days ending at `L` |
| `ta`, `rh`, `stp_tsoil20` | air temperature, humidity, soil temperature at 20 cm on `L` |
| `day_of_year` | of `d` |
| `precip_fc_issue_day`, `precip_fc_target_day`, `precip_fc_gap_sum` | forecast rain for `d−1` and `d` (the two unobserved days), made one day before the issue day |
| `fc_source` | `archive` or `live` |
| `cosmos_vwc` | label `obs(d)`, NaN until observed; `is_labelled` marks rows with a label |
| `inference_ok` | False when `obs(L)` is missing (no forecast issued) |

Split by target date (used in MS3): train 2023–2024, validation 2025, test 2026-01-01 to 2026-10-01, then live
monitoring. **Model selection rule:** the champion is chosen only on validation data (2025 initially, later the
latest 60 labelled days); the 2026 test window is frozen and never used for selection or tuning.

**Missing-day rule.** The pipeline never forward-fills or interpolates. A day missing from the API stays NaN in
`clean/` and is retried by the next `ingest` (every run re-fetches 7 days). A feature that needs a missing day is
NaN (rolling windows need every day). If `obs(L)` itself is missing, no forecast is issued: `inference_ok = False`.

**Archived forecasts, and their limits.** For past dates we use Open-Meteo's *Previous Runs* API, which returns what
the forecast said 1 or 2 days before each hour. That is the information that existed at issue time, unlike the
"Historical Forecast" API, which joins the first hours of each model run and is close to observed rain.
Limits: archive data for BALRD starts around March 2024, so 2023 and early 2024 rows have no forecast features (NaN).
Archived values are reconstructed by Open-Meteo and may differ slightly from what an app showed on the day. Truly
live, as-issued forecasts exist only from the day the `forecast-snapshot` job started (2026-10-30).

## Scheduled workflows

| Workflow | When (UTC) | What |
|---|---|---|
| `ci.yml` | push, pull request | `uv sync --frozen`, `pytest`, `docker build` |
| `forecast_snapshot.yml` | 05:15, 17:15 | store the live forecast as issued |
| `ingest.yml` | 07:30, 19:30 (early guess) | re-fetch last 7 days, upsert, rebuild features |
| `feed_probe.yml` | hourly | is yesterday's record published yet? (temporary) |
| `backfill.yml` | manual | full history + archived forecasts |

All pipeline jobs run inside the Docker image through one reusable workflow (`_pipeline.yml`) and share one
concurrency group, so two jobs never write at the same time.

Storage in Actions: the repository variable `STORAGE_URI` must be `gs://<bucket>/soil-moisture` (jobs fail with a clear
message otherwise), and the secret `GCP_SA_KEY` holds a service-account JSON with Storage Object Admin on the bucket.

## Status

- [x] MS1 proposal ([docs/proposal.pdf](docs/proposal.pdf))
- [x] MS2 feature pipeline (this README, [docs/ms2_summary.pdf](docs/ms2_summary.pdf))
- [ ] MS3 training pipeline (LightGBM, MLflow on DagsHub)
- [ ] MS4 inference pipeline and Streamlit dashboard
