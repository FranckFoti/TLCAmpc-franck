"""CLI: Populate the scenario_runs queue in PostgreSQL.

Usage:
    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.create_queue --runs 20 --n-crit 35
"""

from __future__ import annotations

import argparse

from paper2_tools.distributed.config import COORDINATORS, CONTROLLERS
from paper2_tools.distributed.db import ensure_schema, insert_runs, count_by_status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Populate the distributed scenario queue")
    parser.add_argument("--runs", type=int, required=True, help="Number of runs (0..runs-1)")
    parser.add_argument("--n-crit", type=int, required=True, help="Max drone count offset (drones range: 2..n_crit+1)")
    args = parser.parse_args(argv)

    ensure_schema()

    drones = range(2, args.n_crit + 2)
    runs = range(args.runs)

    inserted = insert_runs(runs, drones, COORDINATORS, CONTROLLERS)
    status_counts = count_by_status()

    print(f"Inserted {inserted} new tasks into queue.")
    print(f"Queue status: {dict(status_counts)}")


if __name__ == "__main__":
    main()
