"""CLI entry point for running ingestion manually."""

import argparse

from ingestion.orchestrator import IngestionOrchestrator
from utils.logging import setup_logger


def main():
    setup_logger("ingestion")
    parser = argparse.ArgumentParser(description="AQI Predictor — Data Ingestion")
    parser.add_argument("--backfill", action="store_true", help="Run historical backfill")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Backfill start date")
    parser.add_argument("--end", type=str, default="2026-07-23", help="Backfill end date")
    parser.add_argument("--fetch", action="store_true", help="Run single fetch cycle")
    args = parser.parse_args()

    orchestrator = IngestionOrchestrator()

    if args.backfill:
        orchestrator.backfill(args.start, args.end)
    elif args.fetch:
        orchestrator.run_full_cycle()
    else:
        # Default: run one cycle
        orchestrator.run_full_cycle()


if __name__ == "__main__":
    main()
