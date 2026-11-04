
= MS2 Summary - Soil Moisture Forecasting
Natalie Sumbo Filipe - https://github.com/sky150/soil-moisture-forecasting-uk

_This project replaces my earlier proposal (urban canopy change in Zürich). The current proposal is
`docs/proposal.pdf`._

- *Station*. Picked the BALRD (Balruddery) station, by checking if there are any gaps. It tied NWYKE (North yorkshire). It has 1,398 daily records, 100 % `cosmos_vwc` coverage, no gaps. Details in _notebook/station_selection.ipynb_.
- *Automated ingestion and backfill.* `backfill` fetches COSMOS-UK daily data for BALRD from 2023-01-01 in yearly
  requests. `ingest` re-fetches the last 7 days each run to pick up provider corrections and retry missing days.
  Both keep the raw API responses unchanged and upsert a clean table keyed by date.
- *Feature engineering in one place.* `src/common/features.py` (`build_feature_rows`) is the only
  feature definition. Training (MS3) and inference (MS4) will import it, which avoids training–serving skew.
  Shifts are by calendar day on a complete daily calendar, so a missing day gives NaN instead of a wrong value.
- *Feature store.* Versioned Parquet (`features/v1/...`) plus a `latest.json` pointer. Storage is addressed by
  `STORAGE_URI` : a local folder or a GCS bucket through the same code path. Inference rows are frozen
  on the issue day (`issued/`) so live evaluation uses what was really known.
- *Scheduled execution.* GitHub Actions: ingest and forecast snapshot twice daily, an hourly arrival probe, a
  manual backfill, and CI (pytest + docker build). Every pipeline job runs inside the Docker image.
- *Reproducibility.* `uv.lock`, Dockerfile, `.env.example`, 21 offline unit tests (including a no-future-data
  test) that need no cloud account, and a README with clone-and-run steps.

Look at first: `common/features.py`, `tests/test_features.py`, `.github/workflows/_pipeline.yml`.

Key decisions: no database or Airflow (one row per day; Actions + Parquet is enough); observations use
"newest fetch wins", forecasts use "first value wins" (keep what was issued); the feature table is rebuilt from
the clean data on every run (~1,400 rows), which makes reruns idempotent.

*Status.* GCS bucket `sky150-soil-moisture` (europe-west6, object versioning on) set up on 30.10.2026.
Scheduled runs on `main` since 02.11.2026.

= Response to reviews
Reviews A and B were written for my earlier proposal and do not apply (see editorial). Persistence, the
hard-to-beat baseline Reviewer B asked for, remains the baseline. Only the editorial points are answered here.

+ *E1 · Fix the station and verify the daily feed* — *Partly.*
  Station fixed to BALRD (`config/settings.yaml`, `notebook/station_selection.ipynb`; commit `c302d2b`).
  Coverage 2023-01-01 to 2026-10-29: 1,398 records, 100 % `cosmos_vwc`, no gaps, 0.2 % missing `precip`.
  The `feed-probe` workflow (`.github/workflows/feed_probe.yml`) checks hourly whether yesterday's record is in the API
  and logs it to `logs/feed_probe.jsonl`. Result over 6 days: yesterday's record usually appeared between about
  05:00 and 10:30 UTC. Once it was late: the record for 01.11 was still missing on 02.11 at 20:40. The 7-day re-fetch
  picked it up later, and no forecast was frozen for 03.11, so the missing-day rule worked as intended.
  GitHub delays scheduled runs, so the "hourly" probe only ran every few hours (17 checks); the measurement is coarse.
  Missing-day rule (README): never forward-fill; a missing day stays NaN and is retried by the next ingest;
  no forecast is issued when `obs(D−1)` is missing (`inference_ok = False`).
  Not yet: the ingest cron is still the early guess (07:30 and 19:30 UTC).
+ *E2 · Add forecast precipitation as an input* — *Partly.*
  New features `precip_fc_issue_day`, `precip_fc_target_day`, `precip_fc_gap_sum`: forecast rain for the two
  unobserved days (`src/common/features.py`, commit `3de7818`). Backfill uses Open-Meteo's Previous Runs API, which
  returns what the forecast said 1 or 2 days ahead (the Historical Forecast API was rejected because it is close to
  observed rain). Archive coverage starts around March 2024, so earlier rows are NaN. Live forecasts are stored exactly
  as issued (`forecast-snapshot`, twice daily, never overwritten). Not yet: MAE split for days with and without rain
  in the gap; this needs the model and is done in MS3.
+ *E3 · State the horizon in one sentence* — *Changed.*
  "A forecast is issued on day D using data through D−1; the target is the daily mean soil moisture of D+1;
  persistence predicts obs(D−1)." Used in the README, `src/common/features.py` and `tests/test_features.py`
  (commit `3de7818`). Persistence is the feature column `vwc_latest`.
+ *E4 · Write down the model-selection rule* — *Changed.*
  The champion is chosen only on validation data (2025 initially, later the latest 60 labelled days); the 2026 test
  window is frozen and never used for selection or tuning. Written in the README (commit `3da430b`);
  the training pipeline (MS3) implements it.
