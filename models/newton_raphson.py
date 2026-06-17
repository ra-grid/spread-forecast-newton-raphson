"""
AR(1) модель: оценка параметров методом Ньютона-Рафсона.

Модель:
    X_t = c + φ·X_{t-1} + ε_t,   ε_t ~ N(0, σ²)

Задача: максимизация функции правдоподобия ⟺ минимизация −log L.
Метод Ньютона-Рафсона итерационно уточняет θ = (c, φ, σ):

    θ_{k+1} = θ_k − H(θ_k)^{-1} · ∇(θ_k)

где:
    ∇  — градиент −log L
    H  — матрица Гессе (вторые производные)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NR_MAX_ITER, NR_TOL


# Квантиль стандартного нормального для 95%-CI (двусторонний)
_Z_95 = 1.959963984540054


@dataclass
class AR1Result:
    c: float       # константа
    phi: float     # коэффициент авторегрессии
    sigma: float   # стандартное отклонение шума
    n_iter: int    # количество итераций
    converged: bool


def _neg_log_likelihood(c: float, phi: float, sigma: float, X: np.ndarray) -> float:
    """−log L для AR(1)."""
    n = len(X) - 1
    residuals = X[1:] - c - phi * X[:-1]
    return n / 2 * np.log(2 * np.pi * sigma ** 2) + np.sum(residuals ** 2) / (2 * sigma ** 2)


def _gradient(c: float, phi: float, sigma: float, X: np.ndarray) -> np.ndarray:
    """
    Градиент −log L по θ = (c, φ, σ).

    ∂/∂c   = −Σ r_t / σ²
    ∂/∂φ   = −Σ r_t · X_{t-1} / σ²
    ∂/∂σ   =  n/σ − Σ r_t² / σ³
    """
    n = len(X) - 1
    r = X[1:] - c - phi * X[:-1]
    X_prev = X[:-1]
    s2 = sigma ** 2

    grad_c = -np.sum(r) / s2
    grad_phi = -np.sum(r * X_prev) / s2
    grad_sigma = n / sigma - np.sum(r ** 2) / (sigma ** 3)
    return np.array([grad_c, grad_phi, grad_sigma])


def _hessian(c: float, phi: float, sigma: float, X: np.ndarray) -> np.ndarray:
    """
    Матрица Гессе −log L по θ = (c, φ, σ).

    H_cc     =  n / σ²
    H_cφ     =  Σ X_{t-1} / σ²
    H_φφ     =  Σ X_{t-1}² / σ²
    H_cσ     =  2 Σ r_t / σ³
    H_φσ     =  2 Σ r_t · X_{t-1} / σ³
    H_σσ     = −n/σ² + 3 Σ r_t² / σ⁴
    """
    n = len(X) - 1
    r = X[1:] - c - phi * X[:-1]
    X_prev = X[:-1]
    s2 = sigma ** 2
    s3 = sigma ** 3
    s4 = sigma ** 4

    H_cc = n / s2
    H_cphi = np.sum(X_prev) / s2
    H_phiphi = np.sum(X_prev ** 2) / s2
    H_csigma = 2 * np.sum(r) / s3
    H_phisigma = 2 * np.sum(r * X_prev) / s3
    H_sigmasigma = -n / s2 + 3 * np.sum(r ** 2) / s4

    H = np.array([
        [H_cc,       H_cphi,      H_csigma],
        [H_cphi,     H_phiphi,    H_phisigma],
        [H_csigma,   H_phisigma,  H_sigmasigma],
    ])
    return H


def estimate_ar1_newton(X: np.ndarray,
                        max_iter: int = NR_MAX_ITER,
                        tol: float = NR_TOL) -> AR1Result:
    """
    Оценка параметров AR(1) методом Ньютона-Рафсона.

    Parameters
    ----------
    X        : временной ряд спреда
    max_iter : максимальное число итераций
    tol      : порог сходимости (‖δθ‖ < tol)

    Returns
    -------
    AR1Result с оценками c, φ, σ
    """
    if len(X) < 3:
        return AR1Result(c=np.nan, phi=np.nan, sigma=np.nan, n_iter=0, converged=False)

    # Начальное приближение: OLS-оценка AR(1)
    X_t = X[1:]
    X_t1 = X[:-1]
    phi_ols = np.cov(X_t, X_t1)[0, 1] / np.var(X_t1) if np.var(X_t1) > 0 else 0.0
    c_ols = np.mean(X_t) - phi_ols * np.mean(X_t1)
    residuals0 = X_t - c_ols - phi_ols * X_t1
    sigma0 = max(np.std(residuals0), 1e-6)

    theta = np.array([c_ols, phi_ols, sigma0])

    for i in range(max_iter):
        c, phi, sigma = theta

        # Защита: σ должна быть > 0
        if sigma <= 0:
            sigma = 1e-6
            theta[2] = sigma

        grad = _gradient(c, phi, sigma, X)
        H = _hessian(c, phi, sigma, X)

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            # Вырожденный Гессе — используем шаг с псевдообратной матрицей
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]

        theta_new = theta - delta

        # σ не может быть <= 0
        theta_new[2] = max(theta_new[2], 1e-6)

        if np.linalg.norm(theta_new - theta) < tol:
            return AR1Result(c=theta_new[0], phi=theta_new[1], sigma=theta_new[2],
                             n_iter=i + 1, converged=True)
        theta = theta_new

    return AR1Result(c=theta[0], phi=theta[1], sigma=theta[2],
                     n_iter=max_iter, converged=False)


def forecast_ar1(result: AR1Result, last_value: float, steps: int) -> np.ndarray:
    """
    Прогноз AR(1):  X̂_{t+k} = c + φ·X̂_{t+k-1}
    (детерминированный, без шума)
    """
    if not result.converged or np.isnan(result.c):
        return np.full(steps, np.nan)

    forecast = [last_value]
    for _ in range(steps):
        forecast.append(result.c + result.phi * forecast[-1])
    return np.array(forecast[1:])


def fitted_values_ar1(result: AR1Result, X: np.ndarray) -> np.ndarray:
    """Подогнанные (in-sample) значения AR(1)."""
    return result.c + result.phi * X[:-1]


# ──────────────────────────────────────────────────────────────
#  Доверительные интервалы из информации Фишера
# ──────────────────────────────────────────────────────────────
#
# Идея.  При оценке параметров θ = (c, φ, σ) методом максимума
# правдоподобия выполняется (асимптотически):
#         θ̂  ~  N( θ,  I(θ)⁻¹ ),
# где I(θ) = E[ ∂²(−log L)/∂θ² ] — информационная матрица Фишера.
# На практике её заменяют наблюдённой информацией — гессианом H(θ̂),
# который мы уже вычисляем в _hessian() на каждом шаге NR.
# Таким образом:
#     Cov(θ̂) ≈ H(θ̂)⁻¹
#     SE(θ̂_i) = √[H(θ̂)⁻¹]_{ii}
#     t_i    = θ̂_i / SE_i
#     95%-CI = θ̂_i ± 1.96 · SE_i
# ──────────────────────────────────────────────────────────────


def parameter_inference_ar1(result: AR1Result, X: np.ndarray) -> Dict[str, dict]:
    """
    Доверительные интервалы и t-статистики параметров AR(1).

    Возвращает словарь:
        {'c':     {'estimate':..,'se':..,'t':..,'ci_low':..,'ci_high':..,'p_value':..},
         'phi':   {...},
         'sigma': {...}}

    p_value считается из двустороннего нормального теста H0: θ_i = 0.
    """
    if not result.converged or np.isnan(result.c):
        return {}

    H = _hessian(result.c, result.phi, result.sigma, X)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    se = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    names = ["c", "phi", "sigma"]
    estimates = np.array([result.c, result.phi, result.sigma])

    out = {}
    for name, est, s in zip(names, estimates, se):
        t = est / s if s > 0 else np.nan
        # двусторонний p-value через нормальное приближение
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
    """CDF стандартного нормального распределения через erf."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def forecast_ar1_with_ci(result: AR1Result,
                         last_value: float,
                         steps: int,
                         alpha: float = 0.05
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Прогноз AR(1) с доверительной полосой уровня (1−alpha).

    Дисперсия прогноза на h шагов (условная относительно последнего наблюдения):
            Var(X̂_{t+h}) = σ² · Σ_{j=0}^{h-1} φ^{2j} = σ² · (1 − φ^{2h}) / (1 − φ²)

    При |φ| < 1 при h → ∞ это сходится к безусловной дисперсии σ²/(1−φ²).

    Returns
    -------
    (mean_forecast, ci_low, ci_high) — массивы длины `steps`.
    """
    if not result.converged or np.isnan(result.c):
        nans = np.full(steps, np.nan)
        return nans, nans, nans

    # Точечный прогноз
    mean = forecast_ar1(result, last_value, steps)

    phi = result.phi
    sigma2 = result.sigma ** 2

    # Var(X̂_{t+h}) = σ² · Σ_{j=0}^{h-1} φ^{2j}
    h = np.arange(1, steps + 1)
    if abs(phi) < 1 - 1e-12:
        var = sigma2 * (1 - phi ** (2 * h)) / (1 - phi ** 2)
    else:
        # |φ|≈1 — нестационарный случай: дисперсия растёт линейно
        var = sigma2 * h
    sd = np.sqrt(var)

    z = _Z_95 if abs(alpha - 0.05) < 1e-12 else _alpha_to_z(alpha)
    return mean, mean - z * sd, mean + z * sd


def _alpha_to_z(alpha: float) -> float:
    """Квантиль 1 − alpha/2 стандартного нормального (через бисекцию по erf)."""
    from math import erf, sqrt
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    # Быстрый тест на синтетическом ряду
    np.random.seed(42)
    c_true, phi_true, sigma_true = 0.001, 0.7, 0.005
    n = 500
    X = [0.0]
    for _ in range(n - 1):
        X.append(c_true + phi_true * X[-1] + np.random.normal(0, sigma_true))
    X = np.array(X)

    res = estimate_ar1_newton(X)
    print(f"Истинные:   c={c_true}, φ={phi_true}, σ={sigma_true}")
    print(f"Оценённые: c={res.c:.6f}, φ={res.phi:.6f}, σ={res.sigma:.6f}")
    print(f"Итераций: {res.n_iter}, сошлось: {res.converged}")
