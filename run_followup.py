"""Follow-up propagation audit CLI (followup/PROTOCOL_FOLLOWUP.md). Stages run one at a time, in order:

    prepare -> smoke -> register -> fit -> validate -> freeze -> predict-test -> evaluate-test -> report
"""
import argparse

from followup import pipeline
from followup.common import load_config

STAGES = ["prepare", "smoke", "register", "fit", "validate", "freeze", "predict-test", "evaluate-test", "report"]
NEEDS = {"register": ["smoke_run"], "validate": ["fit_run"], "freeze": ["fit_run", "val_run"], "report": ["eval_run"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke-run")
    ap.add_argument("--tests-log", help="project-relative pytest log recorded in the registration")
    ap.add_argument("--fit-run")
    ap.add_argument("--val-run")
    ap.add_argument("--eval-run")
    args = ap.parse_args(argv)
    for k in NEEDS.get(args.stage, []):
        if getattr(args, k) is None:
            ap.error(f"{args.stage} needs --{k.replace('_', '-')}")
    cfg = load_config(args.config)
    if args.stage == "prepare":
        run, _ = pipeline.prepare(cfg)
    elif args.stage == "smoke":
        run, _ = pipeline.smoke(cfg)
    elif args.stage == "register":
        reg = pipeline.register(cfg, args.smoke_run, args.tests_log)
        print("registered", reg["created_utc"])
        return
    elif args.stage == "fit":
        run, _ = pipeline.fit(cfg)
    elif args.stage == "validate":
        run, _ = pipeline.validate(cfg, args.fit_run)
    elif args.stage == "freeze":
        fr = pipeline.freeze(cfg, args.fit_run, args.val_run)
        print("frozen; proceed_to_test =", fr["proceed_to_test"])
        return
    elif args.stage == "predict-test":
        run, _ = pipeline.predict_test(cfg)
    elif args.stage == "evaluate-test":
        run, _ = pipeline.evaluate_test(cfg)
    else:
        run, _ = pipeline.report(cfg, args.eval_run)
    print(run.id)


if __name__ == "__main__":
    main()
