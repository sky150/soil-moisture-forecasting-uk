"""Command line for the training pipeline.

    python -m training train                         # newest snapshot of a finished day
    python -m training train --snapshot 2026-11-22   # pin a snapshot
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

from common.config import load_config
from common.storage import Storage
from common.store import FeatureStore
from training import pipeline


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m training")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--snapshot", type=date.fromisoformat, help="YYYY-MM-DD (default: newest finished day)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    fs = FeatureStore(Storage(cfg.storage_uri), cfg.station.id, cfg.features.version)
    logging.info("Station %s, storage %s", cfg.station.id, cfg.storage_uri)

    if args.command == "train":
        pipeline.train(cfg, fs, args.snapshot)


if __name__ == "__main__":
    main()
