import pytest
import torch

import scp  # noqa: F401
from scp.common import load_config
from scp.data import load_pilot, make_batch
from scp.model import load_model
from scp.spectral import svd_w_in


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def cpu_model(cfg):
    torch.manual_seed(cfg["seed"])
    return load_model(cfg, torch.device("cpu"))


@pytest.fixture(scope="session")
def pilot():
    return load_pilot()


@pytest.fixture(scope="session")
def spectrum(cpu_model, cfg):
    model, _ = cpu_model
    return svd_w_in(model.transformer.h[cfg["layer"]].mlp.c_fc.weight)


@pytest.fixture(scope="session")
def small(cpu_model, pilot):
    """Six recipients (mixed lengths, so right padding is exercised) and their natural donors."""
    model, tok = cpu_model
    _, pairs, _ = pilot
    idx = [0, 1, 2, 100, 150, 191]
    rows = [pairs[i] for i in idx]
    pad = tok.eos_token_id
    cpu = torch.device("cpu")
    return {"rec": make_batch([p["rec_ids"] for p in rows], pad, cpu),
            "don": make_batch([p["don_ids"] for p in rows], pad, cpu),
            "rows": rows}
