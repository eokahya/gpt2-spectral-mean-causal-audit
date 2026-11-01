"""Discarded-state response-floor gate: isolated development pilot (DEV_PROTOCOL.md).

Imports the frozen `scp` package read-only (its source tree hash is bound as a dependency) and never writes
outside discarded_response_floor_gate/. CPU thread pools are capped before numpy/torch are imported."""
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import scp  # noqa: E402,F401  (pins the project HF cache + offline mode)

GROOT = Path(__file__).resolve().parent
ROOT = GROOT.parent
