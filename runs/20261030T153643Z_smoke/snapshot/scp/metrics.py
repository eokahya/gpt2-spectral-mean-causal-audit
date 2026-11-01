"""Greater-than score, probability difference, KL, and bootstrap CIs."""
import numpy as np
import torch


def v_rd(r, d, n=100):
    """center_J(1[j > d] - 1[j > r]) over the two-digit year tokens j = 0..n-1. Rows for vectors r, d."""
    j = torch.arange(n, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64).reshape(-1, 1)
    d = torch.as_tensor(d, dtype=torch.float64).reshape(-1, 1)
    v = (j > d).double() - (j > r).double()
    v = v - v.mean(-1, keepdim=True)
    if (v.pow(2).sum(-1) == 0).any():
        raise ValueError("v_rd is zero for some pair (r == d); controls must use the pair's v")
    return v


def t_score(dlogits_J, v):
    """T = <dlogits_J, v> / ||v||^2 (v is centred, so a constant logit shift contributes nothing)."""
    dl = dlogits_J.to(torch.float64)
    return (dl * v).sum(-1) / v.pow(2).sum(-1)


def prob_diff(logits, J_ids, r):
    """PD = sum_{j > r} p_j - sum_{j <= r} p_j (Hanna et al. 2023), p from the full-vocabulary softmax."""
    p = torch.softmax(logits.to(torch.float64), -1)[:, J_ids]
    j = torch.arange(100, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64).reshape(-1, 1)
    return (p * ((j > r).double() - (j <= r).double())).sum(-1)


def kl(logits_p, logits_q):
    """KL(p || q) per row, full vocabulary, float64."""
    lp = torch.log_softmax(logits_p.to(torch.float64), -1)
    lq = torch.log_softmax(logits_q.to(torch.float64), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def centered(x):
    return x - x.mean(-1, keepdim=True)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def bootstrap_ci(stat, arrays, B, seed, clusters=None, ci=0.95):
    """Percentile bootstrap of stat(*arrays) over rows (i.i.d.) or over cluster labels (template clusters).

    Returns (point, lo, hi, n_units)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = stat(*arrays)
    if clusters is None:
        idx_sets = (rng.integers(0, n, n) for _ in range(B))
        n_units = n
    else:
        clusters = np.asarray(clusters)
        labels = np.unique(clusters)
        members = [np.flatnonzero(clusters == c) for c in labels]
        n_units = len(labels)
        idx_sets = (np.concatenate([members[i] for i in rng.integers(0, n_units, n_units)]) for _ in range(B))
    stats = np.array([stat(*(a[idx] for a in arrays)) for idx in idx_sets])
    a = (1 - ci) / 2
    return float(point), float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)), n_units
