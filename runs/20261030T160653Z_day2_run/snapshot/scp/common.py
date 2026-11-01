"""Config, seeding, hashing, environment capture, run directories."""
import contextlib
import datetime as dt
import hashlib
import json
import logging
import platform
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import ROOT

DEFAULT_CONFIG = ROOT / "config" / "pilot_v1.json"
SOURCE_GLOBS = ["scp/*.py", "run.py", "run_day2.py", "tests/*.py"]
SNAPSHOT_GLOBS = SOURCE_GLOBS + ["PROTOCOL.md", "PROTOCOL_AMENDMENT_*.md", "PROTOCOL_DAY2.md", "metadata/*.json", "config/*.json", "manifests/*", "pytest.ini", "requirements*.txt",
                                 "registration/*.json"]


def load_config(path=None):
    path = Path(path or DEFAULT_CONFIG)
    with open(path) as f:
        cfg = json.load(f)
    cfg["_path"] = str(path)
    return cfg


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    files = sorted(p for g in SOURCE_GLOBS for p in ROOT.glob(g))
    per_file = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in per_file.items()).encode()).hexdigest()
    return {"tree_sha256": tree, "files": per_file}


def manifest_hashes():
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT / "manifests").glob("*")) if p.is_file()}


def env_info():
    import scipy
    import transformers
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }


def get_device(name):
    """Return the requested device or raise. Never falls back silently."""
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available; refusing to fall back to CPU silently.")
    return torch.device(name)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device):
    snap = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}  # bytes on macOS
    if device.type == "mps":
        snap["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        snap["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return snap


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


class Run:
    """A run directory under runs/ with a manifest, log file and stage walltimes.

    Failed runs are kept (status=failed, traceback recorded)."""

    def __init__(self, stage, cfg, extra=None):
        self.start = utcnow()
        self.id = f"{self.start:%Y%m%dT%H%M%SZ}_{stage}"
        self.dir = ROOT / "runs" / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.cfg = cfg
        self.walltimes = {}
        self.log = logging.getLogger(self.id)
        self.log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(self.dir / "log.txt"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            self.log.addHandler(h)
        self.manifest = {
            "run_id": self.id,
            "stage": stage,
            "label": cfg.get("label"),
            "protocol_version": cfg.get("protocol_version"),
            "start_utc": self.start.isoformat(),
            "argv": sys.argv,
            "seed": cfg["seed"],
            "config_path": cfg["_path"],
            "config_sha256": sha256_file(cfg["_path"]),
            "protocol_sha256": sha256_file(ROOT / "PROTOCOL.md"),
            "source": source_hashes(),
            "manifests": manifest_hashes(),
            "env": env_info(),
            "model": dict(cfg["model"]),
            "svd": dict(cfg["svd"]),
        }
        if extra:
            self.manifest.update(extra)
        self._snapshot()
        self._write_manifest()

    def _snapshot(self):
        """Copy the exact source/config/protocol/manifests used by this run (hashes alone cannot restore code)."""
        dst = self.dir / "snapshot"
        for p in sorted(p for g in SNAPSHOT_GLOBS for p in ROOT.glob(g) if p.is_file()):
            target = dst / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        self.manifest["snapshot_dir"] = str(dst.relative_to(ROOT))

    def elapsed(self):
        return (utcnow() - self.start).total_seconds()

    def _write_manifest(self):
        with open(self.dir / "run_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    @contextlib.contextmanager
    def timer(self, name):
        t0 = time.perf_counter()
        self.log.info(f"[stage] {name} ...")
        try:
            yield
        finally:
            self.walltimes[name] = time.perf_counter() - t0
            self.log.info(f"[stage] {name} done in {self.walltimes[name]:.2f}s")

    def save_json(self, name, obj):
        with open(self.dir / name, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)

    def finish(self, status="ok", error=None):
        self.manifest["status"] = status
        self.manifest["end_utc"] = utcnow().isoformat()
        self.manifest["walltime_s"] = (utcnow() - self.start).total_seconds()
        self.manifest["stage_walltimes_s"] = self.walltimes
        if error is not None:
            self.manifest["error"] = "".join(traceback.format_exception(error))
        self._write_manifest()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)
