"""CLI (run from the project root):

    .venv/bin/python -m discarded_response_floor_gate_v3 <stage> [options]

Stages, in order: prepare -> verify-protected -> verify-v2 -> smoke -> register -> develop -> power -> report
"""
import argparse
import json

from . import pipeline
from .common import load_config, verify_protected, verify_v2_preservation

STAGES = ["prepare", "verify-protected", "verify-v2", "smoke", "register", "develop", "power", "report"]
NEEDS = {"register": ["smoke_run", "tests_log"], "power": ["dev_run"], "report": ["dev_run"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config")
    ap.add_argument("--smoke-run")
    ap.add_argument("--tests-log", help="project-relative pytest log recorded in the registration")
    ap.add_argument("--dev-run")
    ap.add_argument("--power-run")
    a = ap.parse_args(argv)
    for k in NEEDS.get(a.stage, []):
        if getattr(a, k) is None:
            ap.error(f"{a.stage} needs --{k.replace('_', '-')}")
    cfg = load_config(a.config)
    if a.stage == "verify-protected":
        r = verify_protected()
        print(json.dumps(r, indent=2))
        raise SystemExit(0 if r["ok"] else 1)
    if a.stage == "verify-v2":
        r = verify_v2_preservation(cfg)
        print(json.dumps(r, indent=2))
        raise SystemExit(0 if r["ok"] else 1)
    if a.stage == "register":
        print("registered", pipeline.register(cfg, a.smoke_run, a.tests_log)["created_utc"])
        return
    fn = {"prepare": lambda: pipeline.prepare(cfg), "smoke": lambda: pipeline.smoke(cfg),
          "develop": lambda: pipeline.develop(cfg), "power": lambda: pipeline.power(cfg, a.dev_run),
          "report": lambda: pipeline.report(cfg, a.dev_run, a.power_run)}[a.stage]
    run, _ = fn()
    print(run.id)


if __name__ == "__main__":
    main()
