"""
ARIMA и SARIMAX модели для прогнозирования спреда.

ARIMA(p,d,q) — классическая модель без экзогенных переменных.
SARIMAX(p,d,q)(P,D,Q,s) — с VIX как экзогенной переменной (exog).
Параметры задаются в config.py.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ARIMA_ORDER,
    SARIMAX_ORDER,
    SARIMAX_SEASONAL,
    FORECAST_STEPS,
)


def fit_arima(spread: np.ndarray,
              order: tuple = ARIMA_ORDER) -> object:
    """Обучает ARIMA и возвращает подогнанную модель."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = ARIMA(spread, order=order)
        return model.fit()


def forecast_arima(fitted, steps: int = FORECAST_STEPS) -> np.ndarray:
    """Прогноз ARIMA на steps шагов вперёд."""
    try:
        fc = fitted.forecast(steps=steps)
        return np.asarray(fc)
    except Exception as e:
        print(f"[ARIMA] Ошибка прогноза: {e}")
        return np.full(steps, np.nan)


def fit_sarimax(spread: np.ndarray,
                exog: np.ndarray,
                order: tuple = SARIMAX_ORDER,
                seasonal_order: tuple = SARIMAX_SEASONAL) -> object:
    """
    Обучает SARIMAX с VIX как экзогенной переменной.

    Parameters
    ----------
    spread        : ряд спреда (обучающая выборка)
    exog          : ряд VIX (обучающая выборка, той же длины)
    order         : (p, d, q)
    seasonal_order: (P, D, Q, s)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = SARIMAX(
            spread,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def forecast_sarimax(fitted,
                     exog_future: np.ndarray,
                     steps: int = FORECAST_STEPS) -> np.ndarray:
    """
    Прогноз SARIMAX на steps шагов вперёд.
    exog_future — значения VIX для прогнозного периода.
    """
    try:
        fc = fitted.forecast(steps=steps, exog=exog_future)
        return np.asarray(fc)
    except Exception as e:
        print(f"[SARIMAX] Ошибка прогноза: {e}")
        return np.full(steps, np.nan)


def get_residuals(fitted) -> np.ndarray:
    """Возвращает остатки обученной statsmodels модели."""
    return np.asarray(fitted.resid)


if __name__ == "__main__":
    # Быстрый тест
    np.random.seed(42)
    n = 300
    spread = np.cumsum(np.random.normal(0, 0.005, n))
    vix = 20 + np.random.normal(0, 3, n)

    arima_fit = fit_arima(spread)
    arima_fc = forecast_arima(arima_fit, steps=5)
    print("ARIMA forecast:", arima_fc)

    sarimax_fit = fit_sarimax(spread, vix)
    vix_future = np.full(5, vix[-1])
    sarimax_fc = forecast_sarimax(sarimax_fit, vix_future, steps=5)
    print("SARIMAX forecast:", sarimax_fc)
