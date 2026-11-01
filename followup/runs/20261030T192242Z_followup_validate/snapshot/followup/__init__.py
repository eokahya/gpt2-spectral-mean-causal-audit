"""Follow-up propagation audit (followup/PROTOCOL_FOLLOWUP.md).

Independent of the frozen Day-1..3 pipeline: it imports `scp` read-only and never writes under scp/, tests/,
config/, registration/, manifests/, manifests_day3/ or calibration_day3/. All follow-up config, manifests,
registration files and runs live under followup/.
"""
from pathlib import Path

import scp  # noqa: F401  (pins the project HF cache + offline mode before transformers is imported)

FROOT = Path(__file__).resolve().parent
ROOT = FROOT.parent
