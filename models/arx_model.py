"""
ARX(1) модель с экзогенной переменной (VIX).

Модель:
    X_t = c + φ·X_{t-1} + β·Z_t + ε_t,   ε_t ~ N(0, σ²)

где Z_t — значение VIX в момент t (индекс страха).

Параметры (c, φ, β, σ) оцениваются методом Ньютона-Рафсона
по максимуму функции правдоподобия.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NR_MAX_ITER, NR_TOL

_Z_95 = 1.959963984540054


@dataclass
class ARXResult:
    c: float
    phi: float
    beta: float
    sigma: float
    n_iter: int
    converged: bool


def _residuals(c, phi, beta, X, Z):
    return X[1:] - c - phi * X[:-1] - beta * Z[1:]


def _neg_log_likelihood(theta, X, Z):
    c, phi, beta, sigma = theta
    if sigma <= 0:
        return np.inf
    n = len(X) - 1
    r = _residuals(c, phi, beta, X, Z)
    return n / 2 * np.log(2 * np.pi * sigma ** 2) + np.sum(r ** 2) / (2 * sigma ** 2)


def _gradient(theta, X, Z):
    c, phi, beta, sigma = theta
    n = len(X) - 1
    r = _residuals(c, phi, beta, X, Z)
    X_prev = X[:-1]
    Z_curr = Z[1:]
    s2 = sigma ** 2

    grad_c = -np.sum(r) / s2
    grad_phi = -np.sum(r * X_prev) / s2
    grad_beta = -np.sum(r * Z_curr) / s2
    grad_sigma = n / sigma - np.sum(r ** 2) / (sigma ** 3)
    return np.array([grad_c, grad_phi, grad_beta, grad_sigma])


def _hessian(theta, X, Z):
    c, phi, beta, sigma = theta
    n = len(X) - 1
    r = _residuals(c, phi, beta, X, Z)
    X_prev = X[:-1]
    Z_curr = Z[1:]
    s2 = sigma ** 2
    s3 = sigma ** 3
    s4 = sigma ** 4

    H = np.zeros((4, 4))
    H[0, 0] = n / s2
    H[0, 1] = H[1, 0] = np.sum(X_prev) / s2
    H[0, 2] = H[2, 0] = np.sum(Z_curr) / s2
    H[0, 3] = H[3, 0] = 2 * np.sum(r) / s3
    H[1, 1] = np.sum(X_prev ** 2) / s2
    H[1, 2] = H[2, 1] = np.sum(X_prev * Z_curr) / s2
    H[1, 3] = H[3, 1] = 2 * np.sum(r * X_prev) / s3
    H[2, 2] = np.sum(Z_curr ** 2) / s2
    H[2, 3] = H[3, 2] = 2 * np.sum(r * Z_curr) / s3
    H[3, 3] = -n / s2 + 3 * np.sum(r ** 2) / s4
    return H


def estimate_arx_newton(X: np.ndarray,
                        Z: np.ndarray,
                        max_iter: int = NR_MAX_ITER,
                        tol: float = NR_TOL) -> ARXResult:
    """
    Оценка параметров ARX(1) методом Ньютона-Рафсона.

    Parameters
    ----------
    X : ряд спреда EUR/USD
    Z : ряд VIX (синхронизирован с X)
    """
    assert len(X) == len(Z), "X и Z должны иметь одинаковую длину"

    # Начальное приближение: OLS на расширенной матрице признаков
    n = len(X) - 1
    y = X[1:]
    A = np.column_stack([np.ones(n), X[:-1], Z[1:]])
    try:
        ols = np.linalg.lstsq(A, y, rcond=None)[0]
        c0, phi0, beta0 = ols
    except Exception:
        c0, phi0, beta0 = 0.0, 0.5, 0.0

    r0 = y - A @ np.array([c0, phi0, beta0])
    sigma0 = max(np.std(r0), 1e-6)
    theta = np.array([c0, phi0, beta0, sigma0])

    for i in range(max_iter):
        theta[3] = max(theta[3], 1e-6)
        grad = _gradient(theta, X, Z)
        H = _hessian(theta, X, Z)

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]

        theta_new = theta - delta
        theta_new[3] = max(theta_new[3], 1e-6)

        if np.linalg.norm(theta_new - theta) < tol:
            c, phi, beta, sigma = theta_new
            return ARXResult(c=c, phi=phi, beta=beta, sigma=sigma,
                             n_iter=i + 1, converged=True)
        theta = theta_new

    c, phi, beta, sigma = theta
    return ARXResult(c=c, phi=phi, beta=beta, sigma=sigma,
                     n_iter=max_iter, converged=False)


def forecast_arx(result: ARXResult,
                 last_X: float,
                 Z_future: np.ndarray) -> np.ndarray:
    """
    Прогноз ARX(1) на len(Z_future) шагов вперёд.
    Z_future — прогноз VIX (или последнее известное значение, repeated).
    """
    if not result.converged or np.isnan(result.c):
        return np.full(len(Z_future), np.nan)

    forecast = [last_X]
    for z in Z_future:
        x_next = result.c + result.phi * forecast[-1] + result.beta * z
        forecast.append(x_next)
    return np.array(forecast[1:])


def fitted_values_arx(result: ARXResult, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """In-sample подогнанные значения ARX(1)."""
    return result.c + result.phi * X[:-1] + result.beta * Z[1:]


# ──────────────────────────────────────────────────────────────
#  Доверительные интервалы из информации Фишера
# ──────────────────────────────────────────────────────────────
#  См. подробное обоснование в models/newton_raphson.py.
#  Cov(θ̂) ≈ H(θ̂)⁻¹, где θ = (c, φ, β, σ).
# ──────────────────────────────────────────────────────────────


def parameter_inference_arx(result: ARXResult,
                            X: np.ndarray,
                            Z: np.ndarray) -> Dict[str, dict]:
    """
    SE, t-статистики и 95%-CI для параметров (c, φ, β, σ).

    Для коэффициента β при экзогенной переменной t-статистика отвечает
    на вопрос «значимо ли влияние индекса страха на спред?».
    """
    if not result.converged or np.isnan(result.c):
        return {}

    theta = np.array([result.c, result.phi, result.beta, result.sigma])
    H = _hessian(theta, X, Z)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    se = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    names = ["c", "phi", "beta", "sigma"]
    estimates = theta

    out = {}
    for name, est, s in zip(names, estimates, se):
        t = est / s if s > 0 else np.nan
        p = 2 * (1 - _norm_cdf(abs(t))) if np.isfinite(t) else np.nan
        out[name] = {
            "estimate": float(est),
            "se":       float(s),
            "t":        float(t),
            "ci_low":   float(est - _Z_95 * s),
            "ci_high":  float(est + _Z_95 * s),
            "p_value":  float(p),
        }
    return out


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def forecast_arx_with_ci(result: ARXResult,
                         last_X: float,
                         Z_future: np.ndarray,
                         alpha: float = 0.05
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Прогноз ARX(1) с доверительной полосой (при условии заданного пути Z).

    Условная дисперсия прогноза такая же, как у AR(1):
        Var(X̂_{t+h} | X_t, Z) = σ² · (1 − φ^{2h}) / (1 − φ²)

    Замечание: эта формула предполагает, что будущий путь Z известен точно.
    На практике мы подаём наивный прогноз Z = const, поэтому полоса показывает
    неопределённость *условно* на заданном сценарии Z, а не общую.
    """
    if not result.converged or np.isnan(result.c):
        nans = np.full(len(Z_future), np.nan)
        return nans, nans, nans

    mean = forecast_arx(result, last_X, Z_future)

    phi = result.phi
    sigma2 = result.sigma ** 2
    steps = len(Z_future)
    h = np.arange(1, steps + 1)
    if abs(phi) < 1 - 1e-12:
        var = sigma2 * (1 - phi ** (2 * h)) / (1 - phi ** 2)
    else:
        var = sigma2 * h
    sd = np.sqrt(var)

    return mean, mean - _Z_95 * sd, mean + _Z_95 * sd


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    Z = 20 + 5 * np.random.randn(n)  # синтетический VIX
    X = [0.001]
    for t in range(1, n):
        X.append(0.0005 + 0.65 * X[-1] + 0.0003 * Z[t] + np.random.normal(0, 0.005))
    X = np.array(X)

    res = estimate_arx_newton(X, Z)
    print(f"c={res.c:.6f}, φ={res.phi:.6f}, β={res.beta:.6f}, σ={res.sigma:.6f}")
    print(f"Итераций: {res.n_iter}, сошлось: {res.converged}")
