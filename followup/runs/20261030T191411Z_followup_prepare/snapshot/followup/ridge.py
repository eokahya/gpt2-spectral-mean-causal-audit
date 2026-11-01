"""Standardized ridge on fit rows, deterministic validation lambda selection, fit-mean predictor."""
import numpy as np


def ridge_fit(X, y, lam):
    """Minimise ||y - ybar - Z w||^2 + lam ||w||^2 with Z = (X - mean)/std (fit statistics, ddof=0);
    the intercept ybar is unpenalised."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    mean, std = X.mean(0), X.std(0)
    if (std <= 0).any() or not np.isfinite(std).all():
        raise ValueError(f"degenerate feature column(s) {np.flatnonzero(~(std > 0)).tolist()}")
    Z = (X - mean) / std
    ybar = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ybar))
    return {"lambda": float(lam), "mean": mean.tolist(), "std": std.tolist(), "coef": w.tolist(), "intercept": ybar}


def ridge_predict(model, X, clamp_min=0.0):
    Z = (np.asarray(X, np.float64) - np.array(model["mean"])) / np.array(model["std"])
    return np.maximum(model["intercept"] + Z @ np.array(model["coef"]), clamp_min)


def select_lambda(mse_by_lambda, tie_rel_tol):
    """Smallest validation MSE; within the relative tie tolerance the LARGER lambda wins."""
    lams = sorted(mse_by_lambda)
    best = min(mse_by_lambda.values())
    tied = [lam for lam in lams if mse_by_lambda[lam] - best <= tie_rel_tol * max(abs(best), 1e-300)]
    return max(tied)
