"""Paths, hashes, write-once files, protected-file manifest and run directories."""
import contextlib
import hashlib
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from scp.common import env_info, sha256_file, source_hashes as scp_source_hashes, utcnow

from . import GROOT, ROOT

CONFIG = GROOT / "config" / "gate_dev_v3.json"
REG_DIR = GROOT / "registration"
RUNS = GROOT / "runs"
PROTECTED = GROOT / "protected_manifest.json"
_DEFAULT_PATHS = {"manifest_dir": "manifests", "protocol": "DEV_PROTOCOL_v3.md",
                  "registration": "registration/DEV_REGISTRATION_v3.json"}
# Disjoint from the scp, followup and v2-gate source globs.
SOURCE_GLOBS = ["discarded_response_floor_gate_v3/*.py", "discarded_response_floor_gate_v3/tests/*.py"]
# Old and registered work that must stay byte-identical: everything protected for v2, the WHOLE registered v2 gate
# package (source, configs, protocols, manifests, registration, prepare/smoke/failed-develop runs, logs) and the
# two numerical diagnostic versions with their runs. (Other notes/ files are excluded: the coordinator edits them.)
PROTECTED_GLOBS = ["scp/**/*", "tests/**/*", "config/**/*", "registration/**/*", "manifests/**/*",
                   "manifests_day3/**/*", "calibration_day3/**/*", "followup/**/*", "tests_followup/**/*",
                   "runs/**/*", "metadata/**/*", "run.py", "run_day2.py", "run_day3.py", "run_followup.py",
                   "*.md", "pytest.ini", "requirements*.txt", "discarded_response_floor_gate/**/*",
                   "notes/gate_numerical_diagnostic/diagnose.py", "notes/gate_numerical_diagnostic/diagnose_v1.py",
                   "notes/gate_numerical_diagnostic/test_diagnose.py",
                   "notes/gate_numerical_diagnostic/projector_reconstruction_audit.json",
                   "notes/gate_numerical_diagnostic/runs/**/*"]


def load_config(path=None):
    path = path or CONFIG
    cfg = json.loads(open(path).read())
    cfg["_path"] = str(path)
    return cfg


def rel(p):
    return str(p.relative_to(ROOT))


def _path(cfg, key):
    return GROOT / cfg.get("paths", {}).get(key, _DEFAULT_PATHS[key])


def manifest_dir(cfg):
    return _path(cfg, "manifest_dir")


def protocol_path(cfg):
    return _path(cfg, "protocol")


def registration_path(cfg):
    return _path(cfg, "registration")


def _tree(files):
    per = {rel(p): sha256_file(p) for p in files}
    return {"tree_sha256": hashlib.sha256("".join(f"{k}:{v}\n" for k, v in per.items()).encode()).hexdigest(),
            "files": per}


def source_tree():
    return _tree(sorted(p for g in SOURCE_GLOBS for p in ROOT.glob(g) if p.is_file()))


def manifest_hashes(cfg):
    d = manifest_dir(cfg)
    if not d.exists():
        return {}
    return {rel(p): sha256_file(p) for p in sorted(d.glob("*")) if p.is_file()}


def protected_files():
    out = set()
    for g in PROTECTED_GLOBS:
        for p in ROOT.glob(g):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and ".pytest_cache" not in p.parts \
                    and p.name != ".DS_Store":
                out.add(p)
    return sorted(out)


def protected_hashes():
    return _tree(protected_files())


def verify_protected():
    """Every protected file listed at prepare time must still exist with the same hash, and no new file may appear
    under the protected globs."""
    ref = json.loads(PROTECTED.read_text())
    now = protected_hashes()
    changed = sorted(k for k in ref["files"] if now["files"].get(k) != ref["files"][k])
    added = sorted(set(now["files"]) - set(ref["files"]))
    return {"ok": not changed and not added, "changed_or_missing": changed, "added": added,
            "n_files": len(ref["files"]), "tree_sha256": now["tree_sha256"]}


def verify_v2_preservation(cfg):
    """Every recorded v2 byte must be unchanged: registration file, bound v2 source tree, config, protocol,
    manifests_v2 and the failed development run's manifest/outputs; the v3 manifest copy must equal manifests_v2."""
    pres = cfg["v2_preservation"]
    problems = []
    for path, sha in pres["files"].items():
        p = ROOT / path
        if not p.exists() or sha256_file(p) != sha:
            problems.append(f"changed or missing: {path}")
    v2_src = sorted([*(ROOT / "discarded_response_floor_gate").glob("*.py"),
                     *(ROOT / "discarded_response_floor_gate" / "tests").glob("*.py")])
    if _tree(v2_src)["tree_sha256"] != pres["v2_source_tree_sha256"]:
        problems.append("v2 source tree hash differs from the v2 registration")
    for name in pres["manifest_copy"]:
        a, b = ROOT / pres["v2_manifest_dir"] / name, manifest_dir(cfg) / name
        if not b.exists() or sha256_file(a) != sha256_file(b):
            problems.append(f"v3 manifest copy differs from v2: {name}")
    return {"ok": not problems, "problems": problems}


def bound_hashes(cfg):
    scp_tree = scp_source_hashes()["tree_sha256"]
    proto = protocol_path(cfg)
    return {"protocol_path": rel(proto), "protocol_sha256": sha256_file(proto) if proto.exists() else None,
            "config_path": rel(Path(cfg["_path"]).resolve()), "config_sha256": sha256_file(cfg["_path"]),
            "source": source_tree(), "scp_dependency_tree_sha256": scp_tree,
            "manifest_dir": rel(manifest_dir(cfg)), "manifests": manifest_hashes(cfg),
            "protected_manifest_sha256": sha256_file(PROTECTED) if PROTECTED.exists() else None,
            "artifact_file_sha256": cfg["artifact"]["file_sha256"], "J_ids_sha256": cfg["observable"]["J_ids_sha256"],
            "v2_registration_sha256": cfg["v2_preservation"]["files"][
                "discarded_response_floor_gate/registration/DEV_REGISTRATION_v2.json"]}


def write_once(path, obj):
    if path.exists():
        raise FileExistsError(f"{path} exists; never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, indent=2, default=json_default)


def json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class GRun:
    """A new run directory under discarded_response_floor_gate_v3/runs/. Failed runs are kept with the traceback; on
    finish every output file is hashed into run_manifest.json."""

    def __init__(self, stage, cfg, extra=None):
        self.start = utcnow()
        self.id = f"{self.start:%Y%m%dT%H%M%SZ}_gate_v3_{stage}"
        self.dir = RUNS / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.cfg, self.walltimes = cfg, {}
        self.log = logging.getLogger(self.id)
        self.log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(self.dir / "log.txt"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            self.log.addHandler(h)
        self.manifest = {"run_id": self.id, "stage": stage, "protocol_version": cfg["protocol_version"],
                         "config_path": rel(Path(cfg["_path"]).resolve()), "manifest_dir": rel(manifest_dir(cfg)),
                         "start_utc": self.start.isoformat(), "argv": sys.argv, "bound": bound_hashes(cfg),
                         "env": env_info(), "torch_threads": torch.get_num_threads()}
        if extra:
            self.manifest.update(extra)
        dst = self.dir / "snapshot"
        for g in SOURCE_GLOBS + ["discarded_response_floor_gate_v3/*.md", "discarded_response_floor_gate_v3/config/*.json",
                                 rel(manifest_dir(cfg)) + "/*", "discarded_response_floor_gate_v3/registration/*.json"]:
            for p in sorted(ROOT.glob(g)):
                if p.is_file():
                    t = dst / p.relative_to(ROOT)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, t)
        self.write_manifest()

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

    def finish(self, status="ok", error=None):
        self.manifest.update(status=status, end_utc=utcnow().isoformat(), walltime_s=self.elapsed(),
                             stage_walltimes_s=self.walltimes,
                             outputs={str(p.relative_to(self.dir)): sha256_file(p) for p in sorted(self.dir.glob("*"))
                                      if p.is_file() and p.name not in ("run_manifest.json", "log.txt")})
        if error is not None:
            self.manifest["error"] = "".join(traceback.format_exception(error))
        self.write_manifest()


def load_run(run_id, stage):
    d = RUNS / run_id
    man = json.loads((d / "run_manifest.json").read_text())
    if man.get("stage") != stage or man.get("status") != "ok":
        raise ValueError(f"{run_id}: expected an ok '{stage}' run, got {man.get('stage')}/{man.get('status')}")
    for name, sha in man["outputs"].items():
        if sha256_file(d / name) != sha:
            raise ValueError(f"{run_id}/{name}: output changed after the run")
    return d, man
