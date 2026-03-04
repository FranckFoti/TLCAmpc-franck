"""Shared configuration for the distributed scenario runner."""

from __future__ import annotations

import os

DATABASE_URL_ENV = "DRONESIM_DATABASE_URL"

COORDINATORS = ['mpc_central', 'dmpc_admm']
CONTROLLERS = ['mpc_agent', 'mpc_agent_adaptive']

HORIZON: int = 4
DT: float = 0.1
ALPHA: float = 1.5

MAX_STEPS: int = 10000
TRACE_LEN: int = 10000
TIMEOUT_MS: int = 86400


def get_database_url() -> str:
    url = os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(
            f"Environment variable {DATABASE_URL_ENV} is not set. "
            f"Example: export {DATABASE_URL_ENV}='postgresql://user:pass@host:5432/dronesim'"
        )
    return url
