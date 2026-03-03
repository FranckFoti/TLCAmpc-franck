"""CLI: Export metrics from PostgreSQL to CSV (identical format for plotting).

Usage:
    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.export_csv \
        --output paper2_tools/paper2_results/results_hetzner/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper2_tools.distributed.db import export_metrics_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export metrics from DB to CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args(argv)

    fieldnames, rows = export_metrics_rows()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
