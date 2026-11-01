"""Observable, responses, the within-group statistic U_g, scale and the development summary (CPU float64)."""
import numpy as np
import torch

F64 = torch.float64


def centred_J(logits, J):
    l = logits[..., J].to(F64)
    return l - l.mean(-1, keepdim=True)


def norm2_m(v):
    """||v||_m^2 = sum_j v_j^2 / m over the last axis."""
    return v.pow(2).sum(-1) / v.shape[-1]


def u_variance(R):
    """(1/K) sum_k ||R_k - Rbar||_m^2 (population form; K = 8 enumerated donors)."""
    return float(norm2_m(R - R.mean(0, keepdim=True)).sum() / R.shape[0])


def u_pairs(R):
    """(1/K^2) sum_{i<j} ||R_i - R_j||_m^2 (= (1/64) sum_{i<j} for K = 8)."""
    K = R.shape[0]
    iu = torch.triu_indices(K, K, 1)
    return float(norm2_m(R[iu[0]] - R[iu[1]]).sum() / K ** 2)


def scale_s2(R_nat, w):
    return float((torch.as_tensor(w, dtype=F64) * norm2_m(R_nat)).sum())


def template_T(groups, U, s2):
    """T_ft = mean of the template's group U / s^2; returns {frame: {template: T}}."""
    acc = {}
    for g, u in zip(groups, U):
        acc.setdefault(g["frame"], {}).setdefault(g["template"], []).append(u)
    return {f: {t: float(np.mean(v)) / s2 for t, v in sorted(d.items())} for f, d in sorted(acc.items())}


def frame_moments(T):
    """m_f (mean of the frame's template T) and v_f (sample variance, ddof=1); mbar = mean_f m_f."""
    m = {f: float(np.mean(list(d.values()))) for f, d in T.items()}
    v = {f: float(np.var(list(d.values()), ddof=1)) for f, d in T.items()}
    return m, v, float(np.mean(list(m.values())))
