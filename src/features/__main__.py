"""Command line for the feature pipeline.

    python -m features backfill [--start 2023-01-01] [--end 2026-10-29]
    python -m features ingest
    python -m features forecast-snapshot
    python -m features forecast-backfill [--start ...] [--end ...]
    python -m features build-features
    python -m features probe
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

from common.config import load_config
from common.storage import Storage
from features import pipeline


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m features")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ["backfill", "forecast-backfill"]:
        p = sub.add_parser(name)
        p.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD (default: from settings)")
        p.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD (default: yesterday)")
    for name in ["ingest", "forecast-snapshot", "build-features", "probe"]:
        sub.add_parser(name)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    store = Storage(cfg.storage_uri)
    # Only the URI is logged, never credentials.
    logging.info("Station %s, storage %s", cfg.station.id, cfg.storage_uri)

    if args.command == "backfill":
        pipeline.backfill(cfg, store, args.start, args.end)
    elif args.command == "forecast-backfill":
        pipeline.forecast_backfill(cfg, store, args.start, args.end)
    elif args.command == "ingest":
        pipeline.ingest(cfg, store)
    elif args.command == "forecast-snapshot":
        pipeline.forecast_snapshot(cfg, store)
    elif args.command == "build-features":
        pipeline.build_features(cfg, store)
    elif args.command == "probe":
        pipeline.probe(cfg, store)


if __name__ == "__main__":
    main()
