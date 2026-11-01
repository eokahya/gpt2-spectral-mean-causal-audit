"""Per-row (recipient x alpha) features and labels. Features use only unperturbed forwards, local float64
algebra and the cached probe VJPs; labels need the perturbed forwards and are computed elsewhere."""
import math

import numpy as np
import torch

from .downstream import directional

F64 = torch.float64


def shuffle_perm(d, seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(d, generator=g)


def recipient_unit(rows):
    return [{k: p[k] for k in ("rid", "split", "frame", "noun", "template", "direction", "diff_bin", "r", "d")}
            for p in rows]


def feature_rows(cfg, local, m_B, rows, unp, s, perm):
    """unp: x_r, x_a [N, d], G [N, n, d] (float64 CPU), kl [N]. Returns one dict per (recipient, alpha) row."""
    if not (math.isfinite(s) and s > cfg["evaluation"]["s_floor"]):
        raise ValueError(f"scale s={s} is not a finite value above the floor; evaluation invalid")
    G = unp["G"]
    halves = cfg["probes"]["reliability_split"]
    jnorm = G.pow(2).sum(-1).mean(-1).sqrt()
    d_model = G.shape[-1]
    out = []
    base = recipient_unit(rows)
    for alpha in cfg["intervention"]["alphas"]:
        lq = local.quantities(unp["x_r"], unp["x_a"], alpha, m_B)
        db, D1 = lq["delta_b"], lq["D1"]
        n_q, n_aR = lq["q"].norm(dim=-1), lq["a_R"].norm(dim=-1)
        feats = {
            "alpha": torch.full((len(rows),), float(alpha), dtype=F64),
            "disc_input_norm": lq["disc_input"].norm(dim=-1),
            "delta_norm": lq["delta"].norm(dim=-1),
            "preact_delta_norm": n_aR,
            "disc_preact_norm": n_q,
            "preact_norm_product": n_q * n_aR,
            "b_rec_norm": lq["b_rec"].norm(dim=-1),
            "b_prime_norm": lq["b_prime"].norm(dim=-1),
            "delta_b_norm": db.norm(dim=-1),
            "kl_unpert": unp["kl"],
            "jnorm": jnorm,
            "b_iso": jnorm * db.norm(dim=-1) / (s * math.sqrt(d_model)),
            "dcurv_norm": D1.norm(dim=-1),
            "d_shuf": directional(G, db[:, perm]) / s,
            "d_exact": directional(G, db) / s,
            "d_taylor": directional(G, D1) / s,
            # descriptive only (never a predictor)
            "d_exact_half1": directional(G[:, halves[0]], db) / s,
            "d_exact_half2": directional(G[:, halves[1]], db) / s,
            "a_B_max_abs": lq["a_B"].abs().amax(-1),
        }
        for i, b in enumerate(base):
            out.append({**b, **{k: float(v[i]) for k, v in feats.items()}})
    return out


def design_matrix(rows, names):
    X = np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature value")
    return X


def response_norm(logits_a, logits_b):
    """|| centred(a - b) || per row (float64 CPU)."""
    d = logits_a.to(F64) - logits_b.to(F64)
    return (d - d.mean(-1, keepdim=True)).norm(dim=-1)


def label_norm(red_p, red_b, orig_p, orig_b):
    """|| centred[(red' - red_b) - (orig' - orig_b)] || per row (float64 CPU)."""
    d = (red_p.to(F64) - red_b.to(F64)) - (orig_p.to(F64) - orig_b.to(F64))
    return (d - d.mean(-1, keepdim=True)).norm(dim=-1)


def scale_from_fit(label_rows, alpha=1.0):
    """s = RMS over FIT rows at alpha=1 of the original R-response vector norm. Refuses any other split."""
    splits = {r["split"] for r in label_rows}
    if splits != {"fit"}:
        raise ValueError(f"s must be computed from fit rows only, got splits {sorted(splits)}")
    v = np.array([r["orig_response_norm"] for r in label_rows if r["alpha"] == alpha], dtype=np.float64)
    if len(v) == 0:
        raise ValueError("no fit rows at alpha=1")
    return float(np.sqrt(np.mean(v ** 2)))
