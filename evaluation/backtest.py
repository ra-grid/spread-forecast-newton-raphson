"""
Walk-forward бэктест для всех моделей.

Схема:
  Данные разбиваются на окно обучения и тест.
  Для каждого шага теста:
    1. Обучаем модель на данных до текущей точки.
    2. Прогнозируем на FORECAST_STEPS вперёд.
    3. Сдвигаемся на 1 шаг.

  Итого получаем матрицу прогнозов (n_test × FORECAST_STEPS).
  Метрики усредняются по всем окнам.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEST_SIZE, FORECAST_STEPS
from evaluation.metrics import compute_all


def walk_forward_backtest(
    spread: np.ndarray,
    vix: np.ndarray,
    model_fns: Dict[str, Callable],
    n_test: int = TEST_SIZE,
    forecast_steps: int = FORECAST_STEPS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    spread      : полный ряд спреда
    vix         : полный ряд VIX (та же длина)
    model_fns   : словарь {name: fn}, где fn(spread_train, vix_train, steps) -> np.ndarray
    n_test      : количество точек для тестирования
    forecast_steps : горизонт прогноза

    Returns
    -------
    DataFrame с метриками для каждой модели (строки) и каждого шага горизонта (столбцы).
    """
    n = len(spread)
    train_end = n - n_test

    all_results: List[Dict] = []

    for i in range(n_test):
        train_s = spread[: train_end + i]
        train_v = vix[: train_end + i]

        # Реальные значения на горизонт прогноза
        horizon = min(forecast_steps, n - (train_end + i))
        if horizon <= 0:
            break
        y_true = spread[train_end + i: train_end + i + horizon]

        for name, fn in model_fns.items():
            try:
                y_pred = fn(train_s, train_v, horizon)
                y_pred = np.asarray(y_pred).flatten()[:horizon]
            except Exception as e:
                if verbose:
                    print(f"  [{name}] шаг {i}: ошибка — {e}")
                y_pred = np.full(horizon, np.nan)

            m = compute_all(y_true, y_pred, y_train=train_s)
            m["model"] = name
            m["step"] = i
            all_results.append(m)

        if verbose and i % 5 == 0:
            print(f"  Бэктест: шаг {i + 1}/{n_test}")

    df = pd.DataFrame(all_results)
    return df


def summarize_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Усредняет метрики по всем шагам для каждой модели."""
    numeric_cols = ["MAE", "RMSE", "MASE", "R2", "TheilU"]
    return (
        df.groupby("model")[numeric_cols]
        .mean()
        .round(6)
        .sort_values("RMSE")
    )
