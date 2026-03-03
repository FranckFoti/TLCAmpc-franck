"""Database schema, task claiming, and metrics storage for distributed scenario runs."""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

from paper2_tools.distributed.config import get_database_url

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scenario_runs (
    id              SERIAL PRIMARY KEY,
    run_index       INTEGER NOT NULL,
    n_drones        INTEGER NOT NULL,
    coordinator     TEXT NOT NULL,
    controller      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    claimed_by      TEXT,
    claimed_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    UNIQUE(run_index, n_drones, coordinator, controller)
);

CREATE TABLE IF NOT EXISTS metrics (
    id                      SERIAL PRIMARY KEY,
    scenario_run_id         INTEGER REFERENCES scenario_runs(id),
    num_drones              INTEGER NOT NULL,
    horizon                 INTEGER NOT NULL,
    coordinator             TEXT NOT NULL,
    controller_type         TEXT NOT NULL,
    status                  TEXT NOT NULL,
    steps                   INTEGER NOT NULL,
    wall_time_s             DOUBLE PRECISION,
    min_step_time_s         DOUBLE PRECISION,
    max_step_time_s         DOUBLE PRECISION,
    mean_step_time_s        DOUBLE PRECISION,
    min_distance            DOUBLE PRECISION,
    max_distance            DOUBLE PRECISION,
    mean_distance_all_pairs DOUBLE PRECISION,
    mean_distance_step_mean DOUBLE PRECISION,
    jerk_3d_value           DOUBLE PRECISION,
    created_at              TIMESTAMPTZ DEFAULT now()
);
"""


def get_connection():
    return psycopg2.connect(get_database_url())


def ensure_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Queue population
# ---------------------------------------------------------------------------

def insert_runs(runs: range, drones: range, coordinators: list[str], controllers: list[str]) -> int:
    """Insert pending scenario runs into the queue. Returns number of rows inserted."""
    values = []
    for run_idx in runs:
        for n in drones:
            for coord in coordinators:
                for ctrl in controllers:
                    values.append((run_idx, n, coord, ctrl))

    sql = """
        INSERT INTO scenario_runs (run_index, n_drones, coordinator, controller)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (run_index, n_drones, coordinator, controller) DO NOTHING
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, values, page_size=500)
            inserted = cur.rowcount
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Task claiming  (FOR UPDATE SKIP LOCKED)
# ---------------------------------------------------------------------------

def claim_task() -> dict[str, Any] | None:
    """Atomically claim the next pending task. Returns row dict or None."""
    hostname = socket.gethostname()
    sql = """
        UPDATE scenario_runs
        SET status = 'running', claimed_by = %s, claimed_at = now()
        WHERE id = (
            SELECT id FROM scenario_runs
            WHERE status = 'pending'
            ORDER BY n_drones ASC, id ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, run_index, n_drones, coordinator, controller;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (hostname,))
            row = cur.fetchone()
        conn.commit()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def insert_metrics(scenario_run_id: int, row: dict[str, Any]) -> None:
    """Insert a metrics row linked to the scenario run."""
    sql = """
        INSERT INTO metrics (
            scenario_run_id, num_drones, horizon, coordinator, controller_type,
            status, steps, wall_time_s, min_step_time_s, max_step_time_s,
            mean_step_time_s, min_distance, max_distance, mean_distance_all_pairs,
            mean_distance_step_mean, jerk_3d_value
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s
        )
    """
    params = (
        scenario_run_id,
        row["num_drones"], row["horizon"], row["coordinator"], row["controller_type"],
        row["status"], row["steps"], row["wall_time_s"], row["min_step_time_s"], row["max_step_time_s"],
        row["mean_step_time_s"], row["min_distance"], row["max_distance"], row["mean_distance_all_pairs"],
        row["mean_distance_step_mean"], row["jerk_3d_value"],
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def mark_done(scenario_run_id: int) -> None:
    sql = "UPDATE scenario_runs SET status = 'done', completed_at = now() WHERE id = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (scenario_run_id,))
        conn.commit()


def mark_error(scenario_run_id: int, message: str) -> None:
    sql = "UPDATE scenario_runs SET status = 'error', completed_at = now(), error_message = %s WHERE id = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (message, scenario_run_id))
        conn.commit()


# ---------------------------------------------------------------------------
# Status queries
# ---------------------------------------------------------------------------

def count_by_status() -> dict[str, int]:
    sql = "SELECT status, count(*) FROM scenario_runs GROUP BY status ORDER BY status"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_EXPORT_FIELDNAMES = [
    "num_drones", "horizon", "coordinator", "controller_type", "status", "steps",
    "wall_time_s", "min_step_time_s", "max_step_time_s", "mean_step_time_s",
    "min_distance", "max_distance", "mean_distance_all_pairs",
    "mean_distance_step_mean", "jerk_3d_value",
]


def export_metrics_rows() -> tuple[list[str], list[dict[str, Any]]]:
    """Fetch all completed metrics rows. Returns (fieldnames, rows)."""
    cols = ", ".join(_EXPORT_FIELDNAMES)
    sql = f"SELECT {cols} FROM metrics ORDER BY num_drones, id"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
    return _EXPORT_FIELDNAMES, rows
