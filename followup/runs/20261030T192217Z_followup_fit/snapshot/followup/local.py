"""Exact local quantities of the block-L MLP at the last token, CPU float64.

Row-vector convention (HF Conv1D): z = x @ Wc + b_fc, h = gelu_new(z) @ Wout (+ c_proj bias, which cancels in
every difference used here). P is the frozen PCA column projector; row maps R_row = P^T, Q_row = (I - P)^T.
The frozen bias reduction sees u_hat = (x @ R_row) @ Wc + b_fc + mu, so z = u_hat + q with
q = (x @ Q_row) @ Wc - mu (discarded preactivation minus the frozen calibrated mean).
"""
import math
from dataclasses import dataclass

import torch

from scp.spectral import gelu_new

F64 = torch.float64
_C = math.sqrt(2.0 / math.pi)
_A = 0.044715


def gelu_new_prime(x):
    """Analytic derivative of gelu_new = 0.5 x (1 + tanh(c (x + a x^3)))."""
    t = torch.tanh(_C * (x + _A * x.pow(3)))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _C * (1.0 + 3.0 * _A * x * x)


def taylor_d1(u_hat, a_R, q, a_B, Wout):
    """First-order (in q, a_B) change of orig-minus-red local output between x' and x_r:
    D1 = ([g'(u_hat + a_R) - g'(u_hat)] * q + g'(u_hat + a_R) * a_B) @ Wout  (~ -delta_b)."""
    gp1 = gelu_new_prime(u_hat + a_R)
    return ((gp1 - gelu_new_prime(u_hat)) * q + gp1 * a_B) @ Wout


def exact_delta_b(u_hat, a_R, q, a_B, Wout):
    """delta_b = b(x') - b(x_r), b = h_red - h_orig, written in preactivation coordinates."""
    red = gelu_new(u_hat + a_R) - gelu_new(u_hat)
    orig = gelu_new(u_hat + a_R + q + a_B) - gelu_new(u_hat + q)
    return (red - orig) @ Wout


@dataclass
class LocalMLP:
    Wc: torch.Tensor      # [d, m]
    b_fc: torch.Tensor    # [m]
    Wout: torch.Tensor    # [m, d]
    P: torch.Tensor       # [d, d] column projector (frozen artifact)
    mu: torch.Tensor      # [m] frozen discarded-preactivation mean

    @classmethod
    def from_mlp(cls, mlp, P, mu):
        f = lambda t: t.detach().cpu().to(F64).contiguous()  # noqa: E731
        return cls(f(mlp.c_fc.weight), f(mlp.c_fc.bias), f(mlp.c_proj.weight), f(P), f(mu))

    @property
    def R_row(self):
        return self.P.T

    @property
    def Q_row(self):
        return (torch.eye(self.P.shape[0], dtype=F64) - self.P).T

    def discarded_mean_input(self):
        """m (I - P): the discarded part of the frozen calibration activation mean, recovered from
        mu = m (I-P) Wc. Wc has full row rank, so the solution is unique; returns (m_B, relative residual)."""
        m_B = torch.linalg.lstsq(self.Wc.T, self.mu.unsqueeze(-1)).solution.squeeze(-1)
        res = float((m_B @ self.Wc - self.mu).norm() / self.mu.norm().clamp_min(1e-300))
        return m_B, res

    def preacts(self, x):
        z = x @ self.Wc + self.b_fc
        u_hat = (x @ self.R_row) @ self.Wc + self.b_fc + self.mu
        return z, u_hat, z - u_hat

    def b(self, x):
        z, u_hat, _ = self.preacts(x)
        return (gelu_new(u_hat) - gelu_new(z)) @ self.Wout

    def delta(self, x_r, x_a, alpha):
        """Retained-only intervention delta = alpha * P (x_a - x_r) (row form)."""
        return alpha * ((x_a - x_r) @ self.R_row)

    def quantities(self, x_r, x_a, alpha, m_B):
        """All local per-row quantities for x' = x_r + alpha P (x_a - x_r)."""
        d = self.delta(x_r, x_a, alpha)
        xp = x_r + d
        _, u_hat, q = self.preacts(x_r)
        a_R = (d @ self.R_row) @ self.Wc
        a_B = (d @ self.Q_row) @ self.Wc
        b_rec, b_prime = self.b(x_r), self.b(xp)
        return {"delta": d, "x_prime": xp, "u_hat": u_hat, "q": q, "a_R": a_R, "a_B": a_B,
                "b_rec": b_rec, "b_prime": b_prime, "delta_b": b_prime - b_rec,
                "D1": taylor_d1(u_hat, a_R, q, a_B, self.Wout),
                "disc_input": x_r @ self.Q_row - m_B}
