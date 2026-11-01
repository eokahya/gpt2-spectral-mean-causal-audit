"""Day-2 input-space bases (PROTOCOL_DAY2 §A/§B). All CPU float64, fitted only on the moment-pool inputs X.

Each basis is a *column* map P on the d_model input (x_R = P x, x_B = Q x, Q = I - P). Row-vector code applies
x_row @ P.T. Orthogonal bases have P = P.T; the whitened basis is oblique (P^2 = P, P != P.T in general).
"""
from dataclasses import dataclass, field

import numpy as np
import torch

F64 = torch.float64


@dataclass
class Basis:
    family: str
    k: int
    P: torch.Tensor  # [d, d] column map, float64 CPU
    seed: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def id(self):
        return f"random_s{self.seed}" if self.family == "random" else self.family

    @property
    def Q(self):
        return torch.eye(self.P.shape[0], dtype=F64) - self.P

    def row_maps(self):
        """(row map onto R, row map onto B): delta_row @ R_row, delta_row @ B_row."""
        return self.P.T.contiguous(), self.Q.T.contiguous()


def orth_projector(cols):
    return cols @ cols.T


def centred_cov(X):
    X = X.to(F64)
    m = X.mean(0)
    Xc = X - m
    return Xc.T @ Xc / (X.shape[0] - 1), m


def svd_family(spec):
    return lambda k: Basis("svd", k, orth_projector(spec.V[:, :k]))


def pca_family(X):
    S, _ = centred_cov(X)
    evals, evecs = torch.linalg.eigh(S)
    order = torch.from_numpy(np.argsort(-evals.numpy(), kind="stable"))
    U = evecs[:, order]
    ev = evals[order]
    return lambda k: Basis("pca", k, orth_projector(U[:, :k]),
                           meta={"explained_var_frac": float(ev[:k].clamp_min(0).sum() / ev.clamp_min(0).sum())})


def coord_order(X):
    """Coordinates by E[x_j^2] descending; ties broken by lower index (np.lexsort, last key primary)."""
    m2 = X.to(F64).pow(2).mean(0).numpy()
    return np.lexsort((np.arange(len(m2)), -m2)), m2


def coord_family(X):
    order, m2 = coord_order(X)

    def make(k):
        mask = torch.zeros(len(m2), dtype=F64)
        mask[torch.from_numpy(order[:k].copy())] = 1.0
        return Basis("coord", k, torch.diag(mask), meta={"m2_frac": float(m2[order[:k]].sum() / m2.sum())})
    return make


def haar_orthogonal(d, seed):
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(d, d, generator=g, dtype=F64)
    Qm, R = torch.linalg.qr(A)
    return Qm * torch.sign(torch.diagonal(R))  # sign fix -> Haar-distributed


def random_family(d, seed):
    O = haar_orthogonal(d, seed)
    return lambda k: Basis("random", k, orth_projector(O[:, :k]), seed=seed)


def whitening(X, ridge_rel):
    """C = cov(X) + lam*I with lam = ridge_rel * tr(cov)/d (fixed rule); returns L (Cholesky), L^-1, diagnostics."""
    S, m = centred_cov(X)
    d = S.shape[0]
    lam = ridge_rel * float(torch.trace(S)) / d
    C = S + lam * torch.eye(d, dtype=F64)
    L = torch.linalg.cholesky(C)
    Linv = torch.linalg.solve_triangular(L, torch.eye(d, dtype=F64), upper=False)
    ev_S = torch.linalg.eigvalsh(S)
    ev_C = torch.linalg.eigvalsh(C)
    sv_L = torch.linalg.svdvals(L)
    diag = {"lambda": lam, "ridge_rel": ridge_rel, "n_samples": int(X.shape[0]),
            "eig_cov_min": float(ev_S.min()), "eig_cov_max": float(ev_S.max()),
            "n_eig_cov_below_lambda": int((ev_S < lam).sum()),
            "cond_C": float(ev_C.max() / ev_C.min()), "cond_L": float(sv_L.max() / sv_L.min())}
    return C, L, Linv, m, diag


def wsvd_family(W_in, X, ridge_rel):
    """P = L V'_k V'_k^T L^-1 with W_in L = U' S' V'^T, so W_in P = (W_in L)_k L^-1 (oblique column map)."""
    C, L, Linv, m, diag = whitening(X, ridge_rel)
    U, S, Vh = torch.linalg.svd(W_in.to(F64) @ L, full_matrices=False)
    V = Vh.T

    def make(k):
        P = L @ (V[:, :k] @ V[:, :k].T) @ Linv
        return Basis("wsvd", k, P, meta={**diag, "asym_rel": float((P - P.T).norm() / P.norm())})
    make.parts = {"C": C, "L": L, "Linv": Linv, "U": U, "S": S, "V": V, "mean": m}
    return make


def build_bases(cfg, spec, X_pool_final):
    """All Day-2 bases for all ranks: {(basis_id, k): Basis}. Inputs: Day-1 spectrum, final-token pool X only."""
    bcfg = cfg["bases"]
    fams = []
    for f in bcfg["families"]:
        if f == "svd":
            fams.append(svd_family(spec))
        elif f == "pca":
            fams.append(pca_family(X_pool_final))
        elif f == "coord":
            fams.append(coord_family(X_pool_final))
        elif f == "random":
            fams += [random_family(spec.d_model, s) for s in bcfg["random_seeds"]]
        elif f == "wsvd":
            fams.append(wsvd_family(spec.W_in, X_pool_final, bcfg["ridge_rel"]))
        else:
            raise ValueError(f"unknown basis family {f}")
    out = {}
    for make in fams:
        for k in cfg["ranks"]:
            b = make(k)
            check_projector(b)
            out[(b.id, k)] = b
    return out


def check_projector(b, tol=1e-8):
    """Algebra every basis must satisfy (raises): P^2 = P, P + Q = I, rank k."""
    P = b.P
    err_idem = float((P @ P - P).abs().max())
    err_rank = abs(float(torch.trace(P)) - b.k)
    b.meta.update(idempotence_err=err_idem, trace_err=err_rank)
    if err_idem > tol * max(1.0, float(P.abs().max()) ** 2) or err_rank > 1e-6:
        raise ValueError(f"{b.id} k={b.k}: not a rank-k projector (idem {err_idem:.2e}, trace {err_rank:.2e})")
