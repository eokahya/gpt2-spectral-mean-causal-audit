"""The test split and non-registered layers are refused by the CLI."""
import subprocess
import sys

from scp import ROOT


def _run(*args):
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *args], capture_output=True, text=True, cwd=ROOT)


def test_test_split_is_sealed():
    assert not (ROOT / "registration" / "FROZEN_EPSILON.json").exists()
    p = _run("pilot", "--split", "test")
    assert p.returncode != 0 and "refusing to touch the test split" in p.stderr


def test_only_primary_or_backup_layer():
    p = _run("pilot", "--layer", "5")
    assert p.returncode != 0 and "not allowed" in p.stderr
