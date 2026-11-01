"""Spectral causal pilot: mean/moment compensation of truncated GPT-2 MLP W_in, causal audit.

HF caches are pinned to the project directory and the hub is used offline (model already downloaded).
These env vars must be set before transformers / huggingface_hub are imported.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# No silent CPU fallback for unsupported MPS ops.
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
