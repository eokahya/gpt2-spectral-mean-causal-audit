#!/usr/bin/env python
"""CLI: splits | register | smoke | bench | pilot.

    .venv/bin/python run.py splits      # write manifests/ (tokenizer-validated, seeded)
    .venv/bin/python run.py register    # hash PROTOCOL/config/manifests/source -> registration/REGISTRATION.json
    .venv/bin/python run.py smoke       # original behaviour, positive control, identities, MPS/CPU parity
    .venv/bin/python run.py bench       # 16/32-pair speed + memory
    .venv/bin/python run.py pilot       # exploratory pilot on the calibration split
"""
import argparse
import json
import sys

import scp  # noqa: F401  (sets HF cache env before transformers import)
from scp import ROOT
from scp.common import Run, load_config, manifest_hashes, sha256_file, source_hashes, utcnow

REG = ROOT / "registration" / "REGISTRATION.json"
FROZEN = ROOT / "registration" / "FROZEN_EPSILON.json"


def check_split(split):
    if split == "calibration":
        return
    if split == "test" and not FROZEN.exists():
        sys.exit("refusing to touch the test split: registration/FROZEN_EPSILON.json does not exist")
    sys.exit(f"split {split!r} is not run by this pilot version")


def check_registration(cfg, run):
    if not REG.exists():
        sys.exit("no registration/REGISTRATION.json; run `run.py register` first")
    reg = json.loads(REG.read_text())
    now = {"protocol": sha256_file(ROOT / "PROTOCOL.md"), "config": sha256_file(cfg["_path"]),
           "manifests": manifest_hashes()}
    bad = [k for k in now if reg[k] != now[k]]
    if bad:
        sys.exit(f"registered {bad} changed since registration; bump protocol version")
    if reg["source"]["tree_sha256"] != source_hashes()["tree_sha256"]:
        run.log.warning("source tree changed since registration (recorded in run manifest)")
    run.manifest["registration"] = {"registered_utc": reg["registered_utc"],
                                    "registration_sha256": sha256_file(REG)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["splits", "register", "quadcheck", "smoke", "bench", "pilot"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--layer", type=int, default=None, help="only the fixed backup layer is allowed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    check_split(args.split)
    if args.layer not in (None, cfg["layer"], cfg["backup_layer"]):
        sys.exit(f"layer {args.layer} not allowed; primary {cfg['layer']}, backup {cfg['backup_layer']}")

    if args.stage == "splits":
        from transformers import AutoTokenizer
        from scp.data import write_manifests
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        meta = write_manifests(tok, cfg)
        print(json.dumps({k: meta[k] for k in ("n_pilot_pairs", "n_moment_pool", "n_validation_prompts",
                                               "n_test_prompts")}, indent=1))
        print("excluded by tokenization:", len(meta["excluded_by_tokenization"]))
        return
    if args.stage == "register":
        if REG.exists():
            sys.exit("registration exists; a change requires a new protocol version (not overwriting)")
        REG.parent.mkdir(exist_ok=True)
        reg = {"registered_utc": utcnow().isoformat(), "protocol_version": cfg["protocol_version"],
               "protocol": sha256_file(ROOT / "PROTOCOL.md"), "config": sha256_file(cfg["_path"]),
               "manifests": manifest_hashes(), "source": source_hashes(),
               "note": "written before the first pilot measurement; epsilon NOT frozen; test split sealed"}
        REG.write_text(json.dumps(reg, indent=2))
        print(json.dumps(reg, indent=1))
        return

    from scp import experiment
    run = Run(args.stage + (f"_L{args.layer}" if args.layer else ""), cfg)
    try:
        if args.stage != "quadcheck":  # quadcheck runs before registration (it fixes the node count)
            check_registration(cfg, run)
        ctx = experiment.Context(cfg, run, layer=args.layer)
        if args.stage == "quadcheck":
            res = experiment.quadcheck(ctx)
        elif args.stage == "smoke":
            res = experiment.smoke(ctx)
        elif args.stage == "bench":
            res = experiment.bench(ctx, cfg["budget"]["bench_pairs"])
        else:
            res = experiment.pilot(ctx)
        run.finish("ok")
        run.log.info(f"run dir: {run.dir}")
        return res
    except BaseException as e:
        run.finish("budget_exceeded" if type(e).__name__ == "BudgetExceeded" else "failed", e)
        run.log.error(f"FAILED; raw run kept at {run.dir}")
        raise


if __name__ == "__main__":
    main()
