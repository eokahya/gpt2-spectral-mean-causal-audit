#!/usr/bin/env python
"""Day-3 CLI (PROTOCOL_DAY3.md).

Calibration-only (allowed before registration):
    .venv/bin/python run_day3.py datasets     # held-out pair manifests (no model)
    .venv/bin/python run_day3.py calibrate    # frozen calibration artifact (moment-pool forwards only)
    .venv/bin/python run_day3.py smoke        # artifact == refit, Day-2 replay, calibration gate
After review:
    .venv/bin/python run_day3.py register
    .venv/bin/python run_day3.py validate
    .venv/bin/python run_day3.py freeze --val-run runs/<id>
    .venv/bin/python run_day3.py test
    .venv/bin/python run_day3.py amend --reason "..."   # archive a source change after registration
"""
import argparse
import json
import shutil
import sys

import scp  # noqa: F401
from scp import ROOT
from scp.common import SOURCE_GLOBS, Run, load_config, manifest_hashes, sha256_file, source_hashes, utcnow

CONFIG = ROOT / "config" / "day3_v1.json"


def check_earlier_days_untouched():
    r1 = json.loads((ROOT / "registration" / "REGISTRATION.json").read_text())
    if r1["manifests"] != manifest_hashes() or r1["protocol"] != sha256_file(ROOT / "PROTOCOL.md") \
            or r1["config"] != sha256_file(ROOT / "config" / "pilot_v1.json"):
        sys.exit("Day-1 protocol/config/manifests changed; refusing")
    r2 = json.loads((ROOT / "registration" / "DAY2_REGISTRATION.json").read_text())
    if r2["protocol_day2"] != sha256_file(ROOT / "PROTOCOL_DAY2.md") \
            or r2["config_day2"] != sha256_file(ROOT / "config" / "day2_v1.json"):
        sys.exit("Day-2 protocol/config changed; refusing")


def amend(reason):
    from scp.day3 import AMEND
    tree = source_hashes()["tree_sha256"]
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    arch = ROOT / "registration" / "day3_amendments" / stamp
    arch.mkdir(parents=True, exist_ok=False)
    for p in sorted(p for g in SOURCE_GLOBS + ["PROTOCOL_DAY3.md", "config/day3_v1.json"] for p in ROOT.glob(g)):
        t = arch / p.relative_to(ROOT)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)
    entries = json.loads(AMEND.read_text()) if AMEND.exists() else []
    entries.append({"utc": stamp, "tree_sha256": tree, "reason": reason, "archive": str(arch.relative_to(ROOT))})
    AMEND.write_text(json.dumps(entries, indent=2))
    print(json.dumps(entries[-1], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["datasets", "calibrate", "smoke", "register", "validate", "freeze", "test",
                                      "amend"])
    ap.add_argument("--val-run", default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--split", default=None, help="not accepted: the split is fixed by the stage")
    args = ap.parse_args()
    if args.split is not None:
        sys.exit("--split is not accepted; validate/test stages fix their split")
    cfg = load_config(CONFIG)
    check_earlier_days_untouched()
    from scp import day3

    if args.stage == "datasets":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
        idx = day3.write_datasets(cfg, tok)
        print(json.dumps({s: {k: v for k, v in i.items() if k != "strata"} for s, i in idx["splits"].items()}, indent=1))
        return
    if args.stage == "register":
        print(json.dumps(day3.write_registration(cfg), indent=1))
        return
    if args.stage == "amend":
        if not args.reason:
            sys.exit("--reason required")
        amend(args.reason)
        return
    if args.stage == "freeze":
        if not args.val_run:
            sys.exit("--val-run runs/<id> required")
        s, e = day3.freeze(cfg, args.val_run)
        print(json.dumps({"epsilon": e["epsilon"], "kl_matching": {k: v["claim"] for k, v in
                                                                   s["kl_matching"]["families"].items()}}, indent=1))
        return

    extra = {"day3_protocol_sha256": sha256_file(ROOT / "PROTOCOL_DAY3.md")}
    reg_status, frozen = None, None
    if args.stage in ("validate", "test"):
        try:
            reg, reg_status = day3.verify_registration(cfg)
            if args.stage == "test":
                frozen = day3.verify_frozen(cfg)
        except RuntimeError as e:
            sys.exit(f"refusing {args.stage}: {e}")
        extra["day3_registration"] = {"sha256": sha256_file(day3.REG), "source_status": reg_status}
    run = Run(f"day3_{args.stage}", cfg, extra=extra)
    try:
        from scp.experiment import Context
        art_sha = None
        if args.stage in ("validate", "test"):
            art_sha = reg["bindings"]["calibration_artifact"]
        if args.stage == "calibrate":
            res = day3.build_artifact(Context(cfg, run))
        else:
            from scp.model import model_file_info
            bases, moments, idx = day3.load_artifact(cfg, art_sha, model_file_info(cfg)["safetensors_sha256"])
            art_sha = idx["file_sha256"]
            run.manifest["calibration_artifact_sha256"] = art_sha
            if args.stage == "smoke":
                res = day3.smoke(Context(cfg, run), bases, moments, art_sha)
            else:
                split = "validation" if args.stage == "validate" else "test"
                ctx = day3.HeldoutContext(cfg, run, split)
                if frozen is not None:
                    run.manifest["frozen"] = {"selection_sha256": sha256_file(day3.FROZEN_SEL),
                                              "epsilon_sha256": sha256_file(day3.FROZEN_EPS)}
                res = day3.heldout_run(ctx, bases, moments, art_sha, frozen[0] if frozen else None)
        run.finish("ok")
        run.log.info(f"run dir: {run.dir}")
        return res
    except BaseException as e:
        status = {"BudgetExceeded": "budget_exceeded", "GateInvalid": "invalid_gate"}.get(type(e).__name__, "failed")
        run.finish(status, e)
        run.log.error(f"{status.upper()}; raw run kept at {run.dir}")
        raise


if __name__ == "__main__":
    main()
