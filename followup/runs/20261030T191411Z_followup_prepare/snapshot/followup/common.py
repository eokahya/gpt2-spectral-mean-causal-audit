"""Paths, hashing, write-once files and run directories for the follow-up."""
import contextlib
import hashlib
import json
import logging
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from scp.common import env_info, sha256_file, utcnow

from . import FROOT, ROOT

CONFIG = FROOT / "config" / "followup_v1.json"
PROTOCOL = FROOT / "PROTOCOL_FOLLOWUP.md"
DESIGN = ROOT / "FOLLOWUP_DESIGN.md"
MANIFEST_DIR = FROOT / "manifests"
REG_DIR = FROOT / "registration"
RUNS = FROOT / "runs"
REGISTRATION = REG_DIR / "FOLLOWUP_REGISTRATION.json"
FROZEN = REG_DIR / "FOLLOWUP_FROZEN.json"
TEST_LOCK = REG_DIR / "FOLLOWUP_TEST_PREDICTIONS_LOCK.json"
# Deliberately disjoint from scp.common.SOURCE_GLOBS, so the frozen Day-3 source tree hash is unaffected.
SOURCE_GLOBS = ["followup/*.py", "run_followup.py", "tests_followup/*.py"]


def load_config(path=None):
    path = path or CONFIG
    cfg = json.loads(open(path).read())
    cfg["_path"] = str(path)
    return cfg


def rel(p):
    return str(p.relative_to(ROOT))


def source_tree():
    files = sorted(p for g in SOURCE_GLOBS for p in ROOT.glob(g) if p.is_file())
    per_file = {rel(p): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in per_file.items()).encode()).hexdigest()
    return {"tree_sha256": tree, "files": per_file}


def manifest_hashes():
    if not MANIFEST_DIR.exists():
        return {}
    return {rel(p): sha256_file(p) for p in sorted(MANIFEST_DIR.glob("*")) if p.is_file()}


def bound_hashes(cfg):
    """Everything a registration binds (compared exactly by verify_registration)."""
    return {"protocol_sha256": sha256_file(PROTOCOL) if PROTOCOL.exists() else None,
            "design_sha256": sha256_file(DESIGN), "config_sha256": sha256_file(cfg["_path"]),
            "source": source_tree(), "manifests": manifest_hashes(),
            "day3_config_sha256": sha256_file(ROOT / cfg["artifact"]["day3_config"]),
            "artifact_file_sha256": cfg["artifact"]["file_sha256"]}


def write_once(path, obj):
    """Write JSON to a path that must not exist (registration / freeze / lock files are never overwritten)."""
    if path.exists():
        raise FileExistsError(f"{path} exists; never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, indent=2, default=json_default)


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


class FRun:
    """A follow-up run directory under followup/runs/ (never reused). Failed runs are kept with their traceback;
    on finish every output file is hashed into the run manifest."""

    def __init__(self, stage, cfg, extra=None):
        self.start = utcnow()
        self.id = f"{self.start:%Y%m%dT%H%M%SZ}_followup_{stage}"
        self.dir = RUNS / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.cfg = cfg
        self.walltimes = {}
        self.log = logging.getLogger(self.id)
        self.log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(self.dir / "log.txt"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            self.log.addHandler(h)
        self.manifest = {"run_id": self.id, "stage": stage, "protocol_version": cfg["protocol_version"],
                         "start_utc": self.start.isoformat(), "argv": sys.argv, "seed": cfg["seed"],
                         "config_path": cfg["_path"], "bound": bound_hashes(cfg), "env": env_info(),
                         "model": dict(cfg["model"])}
        if extra:
            self.manifest.update(extra)
        self._snapshot()
        self.write_manifest()

    def _snapshot(self):
        globs = SOURCE_GLOBS + ["followup/PROTOCOL_FOLLOWUP.md", "followup/config/*.json", "followup/manifests/*",
                                "followup/registration/*.json", "FOLLOWUP_DESIGN.md"]
        dst = self.dir / "snapshot"
        for p in sorted(p for g in globs for p in ROOT.glob(g) if p.is_file()):
            t = dst / p.relative_to(ROOT)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)

    def elapsed(self):
        return (utcnow() - self.start).total_seconds()

    def write_manifest(self):
        with open(self.dir / "run_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2, default=json_default)

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
            json.dump(obj, f, indent=2, default=json_default)

    def output_hashes(self):
        return {str(p.relative_to(self.dir)): sha256_file(p) for p in sorted(self.dir.glob("*"))
                if p.is_file() and p.name not in ("run_manifest.json", "log.txt")}

    def finish(self, status="ok", error=None):
        self.manifest.update(status=status, end_utc=utcnow().isoformat(), walltime_s=self.elapsed(),
                             stage_walltimes_s=self.walltimes, outputs=self.output_hashes())
        if error is not None:
            self.manifest["error"] = "".join(traceback.format_exception(error))
        self.write_manifest()


def load_run(run_id, stage):
    """Read a finished run's manifest; it must be the right stage, status ok, with output hashes unchanged."""
    d = RUNS / run_id
    man = json.loads((d / "run_manifest.json").read_text())
    if man.get("stage") != stage or man.get("status") != "ok":
        raise ValueError(f"{run_id}: expected an ok '{stage}' run, got {man.get('stage')}/{man.get('status')}")
    for name, sha in man["outputs"].items():
        if sha256_file(d / name) != sha:
            raise ValueError(f"{run_id}/{name}: output changed after the run")
    return d, man
