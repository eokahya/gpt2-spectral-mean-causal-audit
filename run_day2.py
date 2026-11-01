#!/usr/bin/env python
"""Day-2 CLI (PROTOCOL_DAY2.md): quadcheck | register | smoke | run.

    .venv/bin/python run_day2.py quadcheck   # sigma range of all bases' calibration moments; node-count rule
    .venv/bin/python run_day2.py register    # registration/DAY2_REGISTRATION.json (refuses to overwrite)
    .venv/bin/python run_day2.py smoke       # Day-1 replay (SVD k=192), basis algebra, matching sanity
    .venv/bin/python run_day2.py run         # geometric controls + compensation reference (15-min budget)
"""
import argparse
import json
import sys

import scp  # noqa: F401
from scp import ROOT
from scp.common import Run, load_config, manifest_hashes, sha256_file, source_hashes, utcnow

CONFIG = ROOT / "config" / "day2_v1.json"
REG = ROOT / "registration" / "DAY2_REGISTRATION.json"
DAY1_REG = ROOT / "registration" / "REGISTRATION.json"


def metadata_hashes():
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT / "metadata").glob("*")) if p.is_file()}


def current_hashes(cfg):
    return {"protocol_day2": sha256_file(ROOT / "PROTOCOL_DAY2.md"), "config_day2": sha256_file(cfg["_path"]),
            "manifests": manifest_hashes(), "metadata": metadata_hashes()}


def check_day1_untouched():
    reg = json.loads(DAY1_REG.read_text())
    if reg["manifests"] != manifest_hashes():
        sys.exit("Day-1 manifests changed; refusing")
    if reg["protocol"] != sha256_file(ROOT / "PROTOCOL.md") or reg["config"] != sha256_file(ROOT / "config/pilot_v1.json"):
        sys.exit("Day-1 protocol/config changed; refusing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["quadcheck", "register", "smoke", "run"])
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()
    if args.split != "calibration":
        sys.exit(f"refusing split {args.split!r}: Day 2 runs on the calibration split only")
    cfg = load_config(CONFIG)
    if args.layer not in (None, cfg["layer"]):
        sys.exit(f"layer {args.layer} not allowed on Day 2 (only {cfg['layer']})")
    check_day1_untouched()

    if args.stage == "register":
        if REG.exists():
            sys.exit("Day-2 registration exists; not overwriting")
        if cfg["quadrature_nodes"] is None:
            sys.exit("quadrature nodes not fixed in config; run quadcheck first")
        reg = {"registered_utc": utcnow().isoformat(), "protocol_version": cfg["protocol_version"],
               **current_hashes(cfg), "source": source_hashes(),
               "day1_registration_sha256": sha256_file(DAY1_REG),
               "note": "written before the Day-2 measurement run; validation/test sealed; no epsilon/rank chosen"}
        REG.write_text(json.dumps(reg, indent=2))
        print(json.dumps({k: reg[k] for k in ("registered_utc", "protocol_day2", "config_day2")}, indent=1))
        return

    from scp import day2
    from scp.experiment import Context
    run = Run(f"day2_{args.stage}", cfg, extra={"day2_protocol_sha256": sha256_file(ROOT / "PROTOCOL_DAY2.md"),
                                                "metadata": metadata_hashes()})
    try:
        if args.stage != "quadcheck":
            if not REG.exists():
                sys.exit("no Day-2 registration; run `run_day2.py register` first")
            reg = json.loads(REG.read_text())
            now = current_hashes(cfg)
            bad = [k for k in now if reg[k] != now[k]]
            if bad:
                sys.exit(f"registered {bad} changed since Day-2 registration")
            if reg["source"]["tree_sha256"] != source_hashes()["tree_sha256"]:
                run.log.warning("source tree changed since Day-2 registration (recorded in run manifest)")
            run.manifest["registration"] = {"registered_utc": reg["registered_utc"],
                                            "registration_sha256": sha256_file(REG)}
        ctx = Context(cfg, run)
        res = {"quadcheck": day2.quadcheck, "smoke": day2.smoke, "run": day2.run}[args.stage](ctx)
        run.finish("ok")
        run.log.info(f"run dir: {run.dir}")
        return res
    except BaseException as e:
        run.finish("budget_exceeded" if type(e).__name__ == "BudgetExceeded" else "failed", e)
        run.log.error(f"FAILED; raw run kept at {run.dir}")
        raise


if __name__ == "__main__":
    main()
