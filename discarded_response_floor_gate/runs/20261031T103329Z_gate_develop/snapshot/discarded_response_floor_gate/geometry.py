"""Frozen PCA192 projector and the hybrid MLP-branch inputs (CPU float64, row-vector convention).

Row form of the column-vector definitions: x_minus = P x_rec + (I-P) x_don  <=>  x_rec @ P^T + x_don @ (I-P)^T.
The centred form mu + V z + q with z = V^T(x_rec - mu), q = (I-P)(x_don - mu) equals this for every mu, so no
input mean is needed and none is used."""
import torch

from scp.common import load_config as load_cfg3
from scp.day3 import load_artifact

from . import ROOT

F64 = torch.float64


def projector_checks(P, k, num):
    I = torch.eye(P.shape[0], dtype=F64)
    ev = torch.linalg.eigvalsh((P + P.T) / 2)
    out = {"sym_max_abs": float((P - P.T).abs().max()), "idem_max_abs": float((P @ P - P).abs().max()),
           "trace": float(torch.trace(P)), "n_eig_above_split": int((ev > num["eig_split"]).sum()),
           "eig_retained_min": float(ev[-k:].min()), "eig_discarded_max_abs": float(ev[:-k].abs().max()),
           "complement_idem_max_abs": float(((I - P) @ (I - P) - (I - P)).abs().max())}
    ok = (out["sym_max_abs"] <= num["projector_sym_tol"] and out["idem_max_abs"] <= num["projector_idem_tol"]
          and abs(out["trace"] - k) <= num["projector_trace_tol"] and out["n_eig_above_split"] == k)
    if not ok:
        raise ValueError(f"frozen P fails projector checks: {out}")
    return out


def load_projector(cfg, model_sha=None):
    a = cfg["artifact"]
    cfg3 = load_cfg3(ROOT / a["day3_config"])
    bases, _, _ = load_artifact(cfg3, expected_file_sha=a["file_sha256"], model_sha=model_sha)
    P = bases[(a["basis"], a["k"])].P.to(F64).contiguous()
    return P, projector_checks(P, a["k"], cfg["numerics"])


def hybrids(P, x_rec, x_don, x_a):
    """x_rec, x_a: [d]; x_don: [K, d]. Returns (x_minus [K, d], x_plus [K, d])."""
    R = P.T
    Q = torch.eye(P.shape[0], dtype=F64) - P.T
    x_minus = x_rec @ R + x_don @ Q
    x_plus = x_minus + (x_a - x_rec) @ R
    return x_minus, x_plus


def support(P, x_rec, x_don, x_a, x_minus, x_plus, Wc, b_fc):
    """Mean-free donor-support diagnostics (descriptive only)."""
    R = P.T
    Q = torch.eye(P.shape[0], dtype=F64) - P.T
    dx = x_don - x_rec
    pre = lambda x: x @ Wc + b_fc  # noqa: E731
    return {"retained_distance": (dx @ R).norm(dim=-1), "discarded_displacement": (dx @ Q).norm(dim=-1),
            "donor_discarded_norm_uncentred": (x_don @ Q).norm(dim=-1),
            "recipient_discarded_norm_uncentred": (x_rec @ Q).norm().expand(len(x_don)),
            "x_minus_norm": x_minus.norm(dim=-1), "x_plus_norm": x_plus.norm(dim=-1),
            "x_rec_norm": x_rec.norm().expand(len(x_don)), "x_a_norm": x_a.norm().expand(len(x_don)),
            "preact_minus_norm": pre(x_minus).norm(dim=-1), "preact_plus_norm": pre(x_plus).norm(dim=-1),
            "preact_rec_norm": pre(x_rec).norm().expand(len(x_don)),
            "retained_displacement_norm": ((x_a - x_rec) @ R).norm().expand(len(x_don))}
