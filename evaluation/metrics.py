"""
Метрики качества прогноза.

MAE    — средняя абсолютная ошибка
RMSE   — корень из среднеквадратичной ошибки
MASE   — Mean Absolute Scaled Error (Hyndman & Koehler, 2006)
         Масштаб-инвариантная альтернатива MAPE. Подходит для рядов
         со средним близким к нулю (доходности, спреды), где MAPE
         взрывается из-за деления на малые значения.
         MASE < 1  ⇒  модель лучше наивного прогноза «нет изменения».
R²     — коэффициент детерминации
Theil U — статистика Тейла U1
         < 1  ⇒  модель лучше наивного прогноза (последнее наблюдение).

Замечание: MAPE намеренно убран — для дневных лог-доходов EUR/USD
(среднее ≈ 0) он даёт значения порядка 10⁵ % и непригоден для сравнения.
"""

import numpy as np
from typing import Dict, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray,
         y_pred: np.ndarray,
         y_train: Optional[np.ndarray] = None) -> float:
    """
    MASE = mean(|y - ŷ|) / mean(|Δy_train|)

    Знаменатель — MAE наивного «no-change» прогноза на исторических данных.
    Если y_train не передан, считается на самом y_true (sample-MASE).
    """
    num = np.mean(np.abs(y_true - y_pred))
    if y_train is not None and len(y_train) >= 2:
        denom = np.mean(np.abs(np.diff(y_train)))
    elif len(y_true) >= 2:
        denom = np.mean(np.abs(np.diff(y_true)))
    else:
        return float("nan")
    if denom < 1e-12:
        return float("nan")
    return float(num / denom)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² с защитой от вырожденной дисперсии."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # На горизонте 10 дней ss_tot может быть очень маленькой —
    # порог 1e-12 защищает от деления на ноль и от взрыва R².
    if ss_tot < 1e-12:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Статистика Тейла U1.
    < 1 — модель лучше наивного прогноза (последнее наблюдение).
    """
    if len(y_true) < 2:
        return float("nan")
    naive = y_true[:-1]
    actual = y_true[1:]
    pred = y_pred[1:]
    numerator = np.sqrt(np.mean((actual - pred) ** 2))
    denominator = np.sqrt(np.mean((actual - naive) ** 2))
    if denominator < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def compute_all(y_true: np.ndarray,
                y_pred: np.ndarray,
                y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Возвращает все метрики в виде словаря.

    y_train (опционально) используется для расчёта MASE
    относительно in-sample наивного прогноза.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {k: float("nan") for k in ("MAE", "RMSE", "MASE", "R2", "TheilU")}
    return {
        "MAE":    mae(yt, yp),
        "RMSE":   rmse(yt, yp),
        "MASE":   mase(yt, yp, y_train),
        "R2":     r2(yt, yp),
        "TheilU": theil_u(yt, yp),
    }


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    line = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())
    print(f"{name:20s}: {line}")
