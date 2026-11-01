"""Intervention / replacement semantics on the real GPT-2 weights (CPU float32), plus a small MPS parity check."""
import pytest
import torch

from scp.data import make_batch
from scp.model import SpectralMLP, forward_last, load_model, make_reduction, wrapped_mlp
from scp.spectral import calibrated_moments, weights_only_moments

METHODS = ("zero", "bias", "gauss", "gauss_wo")
SCOPES = ("last_token", "full_layer")


def _moments(cpu_model, cfg, spectrum, k, x):
    model, _ = cpu_model
    mlp = model.transformer.h[cfg["layer"]].mlp
    mlp = getattr(mlp, "orig", mlp)  # unwrap SpectralMLP if installed
    ln = model.transformer.h[cfg["layer"]].ln_2
    P_R, P_B = spectrum.projectors(k)
    Wc = mlp.c_fc.weight.detach()
    cal = calibrated_moments(x, P_B, Wc)
    wo = weights_only_moments(P_B, Wc, ln.weight, ln.bias)
    return P_R, P_B, mlp, cal, wo


def _red(cfg, P_R, mlp, cal, wo, method, scope):
    mu, sd = wo if method == "gauss_wo" else cal
    return make_reduction(P_R, mlp, mu, sd, method, scope, cfg["quadrature_nodes"], cfg["quadrature_sigma_max_checked"],
                          torch.device("cpu"))


def _orig(model, w, small):
    return forward_last(model, w, small["rec"]), forward_last(model, w, small["don"])


def test_last_position_logits_match_full_model(cpu_model, small):
    model, _ = cpu_model
    b = small["rec"]
    with torch.no_grad():
        full = model(input_ids=b.ids, attention_mask=b.mask).logits
    ar = torch.arange(b.ids.shape[0])
    with wrapped_mlp(model, 9) as w:
        res = forward_last(model, w, b)
    assert torch.allclose(res["logits"], full[ar, b.last_idx], atol=1e-4)
    # right padding does not change real-token logits: compare with unpadded single runs
    for i, p in enumerate(small["rows"]):
        with torch.no_grad():
            single = model(input_ids=torch.tensor([p["rec_ids"]])).logits[0, -1]
        assert torch.allclose(single, res["logits"][i], atol=1e-3)


def test_hook_removal_restores_model(cpu_model, small):
    model, _ = cpu_model
    orig_mlp = model.transformer.h[9].mlp
    with torch.no_grad():
        before = model(input_ids=small["rec"].ids, attention_mask=small["rec"].mask).logits
    with wrapped_mlp(model, 9) as w:
        assert isinstance(model.transformer.h[9].mlp, SpectralMLP)
        forward_last(model, w, small["rec"], delta=torch.randn(6, 768))
        with pytest.raises(RuntimeError):
            with wrapped_mlp(model, 9):
                pass
    assert model.transformer.h[9].mlp is orig_mlp
    with pytest.raises(ValueError):
        with wrapped_mlp(model, 9):
            raise ValueError("boom")
    assert model.transformer.h[9].mlp is orig_mlp
    with torch.no_grad():
        after = model(input_ids=small["rec"].ids, attention_mask=small["rec"].mask).logits
    assert torch.equal(before, after)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_same_example_identity(cpu_model, small):
    model, _ = cpu_model
    with wrapped_mlp(model, 9) as w:
        rec, _ = _orig(model, w, small)
        same = forward_last(model, w, small["rec"], delta=torch.zeros(6, 768))
    assert torch.equal(same["logits"], rec["logits"])


def test_residual_bypass_invariant_and_locality(cpu_model, small):
    """Patch touches only the MLP-branch input at the final position; residual bypass and other positions fixed."""
    model, _ = cpu_model
    b = small["rec"]
    cap = {}
    blk = model.transformer.h[9]
    h1 = blk.ln_2.register_forward_pre_hook(lambda m, a: cap.setdefault("resid_mid", []).append(a[0].clone()))
    h2 = blk.register_forward_hook(lambda m, a, o: cap.setdefault("block_out", []).append(
        (o[0] if isinstance(o, tuple) else o).clone()))
    try:
        with wrapped_mlp(model, 9) as w:
            base = forward_last(model, w, b)
            delta = torch.randn(6, 768) * 2.0
            pat = forward_last(model, w, b, delta=delta)
    finally:
        h1.remove(), h2.remove()
    assert torch.equal(cap["resid_mid"][0], cap["resid_mid"][1])  # residual bypass unchanged
    d_block = cap["block_out"][1] - cap["block_out"][0]
    d_mlp = pat["out_all"] - base["out_all"]
    assert torch.allclose(d_block, d_mlp, atol=1e-4)  # block-output change == MLP-output change
    ar = torch.arange(6)
    mask = torch.zeros(d_mlp.shape[:2], dtype=torch.bool)
    mask[ar, b.last_idx] = True
    assert d_mlp[~mask].abs().max() == 0  # other positions untouched
    assert d_mlp[mask].abs().max() > 1e-2
    # final hidden states at earlier *real* positions unchanged (causal); final position changed.
    # (Right-pad positions after the final token may attend to it and are excluded; they are never read.)
    real = b.mask.bool()
    dh = (pat["hidden_all"] - base["hidden_all"]).abs()
    assert dh[real & ~mask].max() == 0 and dh[mask].max() > 1e-3
    assert torch.equal(pat["x_pre_last"], base["x_pre_last"])  # cache of the recipient not mutated


def test_RB_patch_equals_full_donor_branch(cpu_model, small, spectrum):
    model, _ = cpu_model
    with wrapped_mlp(model, 9) as w:
        rec, don = _orig(model, w, small)
        d = don["x_pre_last"] - rec["x_pre_last"]
        P_R, P_B = spectrum.projectors(192)
        d_sum = (d.double() @ P_R + d.double() @ P_B).float()
        assert torch.allclose(d_sum, d, atol=1e-5)
        full = forward_last(model, w, small["rec"], delta=d)
    assert torch.allclose(full["x_last"], don["x_pre_last"], atol=1e-5)
    assert torch.allclose(full["out_last"], don["out_last"], atol=1e-4)
    # independent implementation: overwrite the MLP input with the donor input via a raw pre-hook
    ar = torch.arange(6)
    li = small["rec"].last_idx

    def sub(m, args):
        x = args[0].clone()
        x[ar, li] = don["x_pre_last"]
        return (x,)
    h = model.transformer.h[9].mlp.register_forward_pre_hook(sub)
    try:
        with torch.no_grad():
            hs = model.transformer(input_ids=small["rec"].ids, attention_mask=small["rec"].mask).last_hidden_state
            ref = model.lm_head(hs[ar, li])
    finally:
        h.remove()
    assert torch.allclose(full["logits"], ref, atol=1e-4)


def test_full_rank_reduction_is_original(cpu_model, cfg, spectrum, small):
    model, _ = cpu_model
    with wrapped_mlp(model, 9) as w:
        rec, don = _orig(model, w, small)
        x = torch.cat([rec["x_pre_last"], don["x_pre_last"]])
        P_R, P_B, mlp, cal, wo = _moments(cpu_model, cfg, spectrum, 768, x)
        for method in METHODS:
            for scope in SCOPES:
                red = _red(cfg, P_R, mlp, cal, wo, method, scope)
                assert float(red.sigma.abs().max()) < 1e-6
                out = forward_last(model, w, small["rec"], reduction=red)
                assert torch.allclose(out["logits"], rec["logits"], atol=2e-4), (method, scope)


def test_reduced_B_invariance(cpu_model, cfg, spectrum, small):
    """In every reduced model a patch confined to the discarded subspace B has no effect (Delta_B = 0)."""
    model, _ = cpu_model
    with wrapped_mlp(model, 9) as w:
        rec, don = _orig(model, w, small)
        d = don["x_pre_last"] - rec["x_pre_last"]
        for k in cfg["ranks"]:
            P_R, P_B, mlp, cal, wo = _moments(cpu_model, cfg, spectrum, k, torch.cat([rec["x_pre_last"],
                                                                                    don["x_pre_last"]]))
            dB = (d.double() @ P_B).float()
            o_b = forward_last(model, w, small["rec"], delta=dB)
            assert (o_b["logits"] - rec["logits"]).abs().max() > 1e-2  # nontrivial in the original model
            for method in METHODS:
                for scope in SCOPES:
                    red = _red(cfg, P_R, mlp, cal, wo, method, scope)
                    base = forward_last(model, w, small["rec"], reduction=red)
                    pat = forward_last(model, w, small["rec"], delta=dB, reduction=red)
                    assert (pat["out_last"] - base["out_last"]).abs().max() < 1e-4, (k, method, scope)
                    assert (pat["logits"] - base["logits"]).abs().max() < 1e-3, (k, method, scope)


def test_scope_locality(cpu_model, cfg, spectrum, small):
    model, _ = cpu_model
    with wrapped_mlp(model, 9) as w:
        rec, don = _orig(model, w, small)
        P_R, P_B, mlp, cal, wo = _moments(cpu_model, cfg, spectrum, 96, torch.cat([rec["x_pre_last"],
                                                                                 don["x_pre_last"]]))
        ar = torch.arange(6)
        mask = torch.zeros(rec["out_all"].shape[:2], dtype=torch.bool)
        mask[ar, small["rec"].last_idx] = True
        real = small["rec"].mask.bool()
        lt = forward_last(model, w, small["rec"], reduction=_red(cfg, P_R, mlp, cal, wo, "gauss", "last_token"))
        fl = forward_last(model, w, small["rec"], reduction=_red(cfg, P_R, mlp, cal, wo, "gauss", "full_layer"))
    d_lt = (lt["out_all"] - rec["out_all"]).abs()
    d_fl = (fl["out_all"] - rec["out_all"]).abs()
    assert d_lt[~mask].max() == 0 and d_lt[mask].max() > 1e-3
    assert d_fl[real & ~mask].max() > 1e-3
    # with IDENTICAL moments the final-position MLP output is identical in both scopes (position-wise MLP)
    assert torch.allclose(lt["out_last"], fl["out_last"], atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_cpu_parity(cfg, cpu_model, spectrum, small):
    model_c, _ = cpu_model
    model_m, tok = load_model(cfg, torch.device("mps"))
    rows = [p["rec_ids"] for p in small["rows"]]
    res = {}
    for name, model, dev in (("cpu", model_c, torch.device("cpu")), ("mps", model_m, torch.device("mps"))):
        b = make_batch(rows, tok.eos_token_id, dev)
        mlp = model.transformer.h[9].mlp
        P_R, P_B = spectrum.projectors(192)
        mu, sd = weights_only_moments(P_B, mlp.c_fc.weight.detach().cpu(), model_c.transformer.h[9].ln_2.weight,
                                      model_c.transformer.h[9].ln_2.bias)
        red = make_reduction(P_R, mlp, mu, sd, "gauss", "full_layer", cfg["quadrature_nodes"],
                             cfg["quadrature_sigma_max_checked"], dev)
        with wrapped_mlp(model, 9) as w:
            res[name] = [forward_last(model, w, b)["logits"].cpu(),
                         forward_last(model, w, b, reduction=red)["logits"].cpu()]
    for a, c in zip(res["cpu"], res["mps"]):
        assert (a - c).abs().max() < 2e-3
