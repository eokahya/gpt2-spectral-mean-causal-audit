"""Discarded-state response-floor gate, v3: development TECHNICAL REMEASUREMENT (DEV_PROTOCOL_v3.md).

Self-contained copy of the v2 modules with a canonical no-padding extraction layout. It imports only the frozen
`scp` package (source tree bound as a dependency); it never imports or writes into the registered v2 package
discarded_response_floor_gate/, whose bytes are bound by the v3 protection manifest and the v2 preservation block.
CPU thread pools are capped before numpy/torch are imported."""
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import scp  # noqa: E402,F401  (pins the project HF cache + offline mode)

GROOT = Path(__file__).resolve().parent
ROOT = GROOT.parent
