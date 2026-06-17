"""
Все функции визуализации проекта.
Каждая функция сохраняет график в results/ и опционально показывает его.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, DATA_SOURCE

# ──────────────────────────────────────────────
#  Динамические подписи под выбранный источник
# ──────────────────────────────────────────────
# Назначение поля 'vix' в DataFrame зависит от источника:
#   yahoo → реальный VIX (волатильность S&P 500)
#   moex  → RVI (Russian Volatility Index)
# Аналогично сам торгуемый актив отличается.
if DATA_SOURCE == "moex":
    from config import MOEX_CURRENCY, MOEX_INDEX
    _ASSET = MOEX_CURRENCY.replace("_TOM", "").replace("000UTSTOM", "/USD")
    # человекочитаемое название пары
    _ASSET_PRETTY = {
        "USD000UTSTOM": "USD/RUB",
        "EUR_RUB__TOM": "EUR/RUB",
        "CNYRUB_TOM":   "CNY/RUB",
    }.get(MOEX_CURRENCY, MOEX_CURRENCY)
    ASSET_LABEL = _ASSET_PRETTY
    VOL_LABEL = MOEX_INDEX            # "RVI"
    VOL_LONG = "Russian Volatility Index (RVI)"
else:
    ASSET_LABEL = "EUR/USD"
    VOL_LABEL = "VIX"
    VOL_LONG = "Индекс волатильности VIX"

SPREAD_LABEL = f"Спред (лог-доход {ASSET_LABEL})"


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,        # PNG-файлы остаются с высоким DPI
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    # Чуть больше фигуры по умолчанию — комфортнее зумить в окне.
    "figure.figsize": (15, 7),
})

_COLORS = {
    "AR1":       "#1f77b4",
    "ARX":       "#ff7f0e",
    "ARIMA":     "#2ca02c",
    "SARIMAX":   "#d62728",
    "LSTM-uni":  "#9467bd",
    "LSTM-multi":"#8c564b",
    "actual":    "#000000",
    "train":     "#aec7e8",
    "vix":       "#e377c2",
}


def _save(fig: plt.Figure, name: str, show: bool) -> None:
    """
    Сохраняет фигуру в results/ и опционально оставляет окно открытым.

    При show=True окно НЕ закрывается — все окна копятся, и в конце
    main.py вызывает show_all(), которая блокирующе показывает их разом.
    Это позволяет переключаться между графиками и пользоваться
    панелью matplotlib (зум, пан, экспорт).
    """
    path = RESULTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"[plot] Сохранено: {path}")
    if not show:
        plt.close(fig)


def show_all() -> None:
    """
    Открыть все ранее накопленные фигуры одним блокирующим вызовом.
    Вызывайте в конце main.py при флаге --show.
    """
    if plt.get_fignums():
        print("[plot] Открываю все окна (закройте их, чтобы продолжить)...")
        plt.show()


# ──────────────────────────────────────────────
#  1. Исторический спред + VIX
# ──────────────────────────────────────────────

def plot_spread_and_vix(df: pd.DataFrame, show: bool = False) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df.index, df["spread"], color=_COLORS["train"],
                 linewidth=0.8, label=f"Спред {ASSET_LABEL}")
    axes[0].set_ylabel(SPREAD_LABEL)
    axes[0].set_title(f"Спред {ASSET_LABEL} и {VOL_LONG}")
    axes[0].legend()

    axes[1].plot(df.index, df["vix"], color=_COLORS["vix"],
                 linewidth=0.8, label=VOL_LABEL)
    axes[1].set_ylabel(VOL_LABEL)
    axes[1].set_xlabel("Дата")
    axes[1].legend()

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    fig.tight_layout()
    _save(fig, "01_spread_and_vix", show)


# ──────────────────────────────────────────────
#  2. Корреляция спред ~ VIX
# ──────────────────────────────────────────────

def plot_spread_vix_correlation(df: pd.DataFrame, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["vix"], df["spread"], alpha=0.3, s=8, color=_COLORS["vix"])

    # Линия тренда
    z = np.polyfit(df["vix"], df["spread"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["vix"].min(), df["vix"].max(), 200)
    ax.plot(x_line, p(x_line), color="red", linewidth=1.5,
            label=f"Тренд: y={z[0]:.5f}x+{z[1]:.5f}")

    corr = df[["spread", "vix"]].corr().iloc[0, 1]
    ax.set_title(f"Корреляция спреда {ASSET_LABEL} и {VOL_LABEL} (r = {corr:.3f})")
    ax.set_xlabel(VOL_LABEL)
    ax.set_ylabel(SPREAD_LABEL)
    ax.legend()
    fig.tight_layout()
    _save(fig, "02_spread_vix_corr", show)


# ──────────────────────────────────────────────
#  3. Сравнение прогнозов моделей
# ──────────────────────────────────────────────

def plot_forecasts(
    dates_train: pd.DatetimeIndex,
    spread_train: np.ndarray,
    dates_test: pd.DatetimeIndex,
    spread_test: np.ndarray,
    forecasts: dict,                # {model_name: np.ndarray}
    forecasts_ci: dict | None = None,  # {model_name: (lower, upper)}
    show: bool = False,
    title: str = "Сравнение прогнозов моделей",
    fname: str = "03_forecasts",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    # Контекст: последние N дней обучающей выборки + весь тест.
    # Показывать всю train историю не нужно — на 1800 точек прогноз
    # на 10 дней не виден; берём хвост train в ту же длину, что и тест,
    # чтобы был визуальный «разгон» перед прогнозом.
    ctx = len(dates_test)
    ax.plot(dates_train[-ctx:], spread_train[-ctx:],
            color=_COLORS["train"], linewidth=1.0, alpha=0.8,
            label="Обучающая выборка (хвост)")
    ax.plot(dates_test, spread_test, color=_COLORS["actual"],
            linewidth=1.5, label="Факт (тест)", zorder=5)

    # Длина прогноза (часто меньше длины test, например 10 vs 30)
    horizon_len = 0
    forecasts_ci = forecasts_ci or {}
    for name, fc in forecasts.items():
        color = _COLORS.get(name, None)
        fc_arr = np.asarray(fc).flatten()
        n = min(len(dates_test), len(fc_arr))
        horizon_len = max(horizon_len, n)
        ax.plot(dates_test[:n], fc_arr[:n], linestyle="--", linewidth=1.5,
                color=color, label=name, marker="o", markersize=3)

        # Доверительная полоса 95% — если предоставлена для этой модели
        if name in forecasts_ci:
            lo, hi = forecasts_ci[name]
            lo = np.asarray(lo).flatten()[:n]
            hi = np.asarray(hi).flatten()[:n]
            ax.fill_between(dates_test[:n], lo, hi,
                            color=color, alpha=0.12, linewidth=0,
                            label=f"{name} 95% CI")

    # Визуально подсветить окно, на котором реально считается прогноз
    if 0 < horizon_len < len(dates_test):
        end_of_horizon = dates_test[horizon_len - 1]
        ax.axvspan(dates_test[0], end_of_horizon,
                   color="yellow", alpha=0.07, zorder=0,
                   label=f"Горизонт прогноза ({horizon_len} дн.)")
        ax.axvline(end_of_horizon, color="gray", linestyle=":",
                   linewidth=1.0, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel(SPREAD_LABEL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(loc="upper left", framealpha=0.9)
    plt.xticks(rotation=45)
    fig.tight_layout()
    _save(fig, fname, show)


# ──────────────────────────────────────────────
#  4. Бэктест: метрики по моделям
# ──────────────────────────────────────────────

def plot_backtest_metrics(summary: pd.DataFrame, show: bool = False) -> None:
    metrics = ["MAE", "RMSE", "MASE"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))

    for ax, metric in zip(axes, metrics):
        values = summary[metric].dropna()
        bars = ax.bar(values.index, values.values,
                      color=[_COLORS.get(n, "#888888") for n in values.index])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Средние метрики (walk-forward backtest)", fontsize=13)
    fig.tight_layout()
    _save(fig, "04_backtest_metrics", show)


# ──────────────────────────────────────────────
#  5. LSTM: кривые потерь (train / val)
# ──────────────────────────────────────────────

def plot_lstm_loss(history, model_name: str = "LSTM", show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss")
    ax.set_title(f"{model_name}: динамика потерь (MSE)")
    ax.set_xlabel("Эпохи")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"05_lstm_loss_{model_name.replace(' ', '_')}", show)


# ──────────────────────────────────────────────
#  6. Newton-Raphson: сходимость параметров
# ──────────────────────────────────────────────

def plot_nr_convergence(
    param_history: list,   # список [{'c':..,'phi':..,'sigma':..}] по итерациям
    show: bool = False,
) -> None:
    df = pd.DataFrame(param_history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = {"c": "c (константа)", "phi": "φ (авторегрессия)", "sigma": "σ (шум)"}
    for ax, col in zip(axes, ["c", "phi", "sigma"]):
        ax.plot(df[col])
        ax.set_title(f"Сходимость {labels[col]}")
        ax.set_xlabel("Итерация")
        ax.set_ylabel(col)
    fig.suptitle("Метод Ньютона-Рафсона: сходимость AR(1)", fontsize=12)
    fig.tight_layout()
    _save(fig, "06_nr_convergence", show)


# ──────────────────────────────────────────────
#  7. Остатки (residuals) моделей
# ──────────────────────────────────────────────

def plot_residuals(
    dates: pd.DatetimeIndex,
    residuals: dict,   # {name: array}
    show: bool = False,
) -> None:
    n = len(residuals)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, residuals.items()):
        color = _COLORS.get(name, "#888888")
        m = min(len(dates), len(res))
        ax.plot(dates[:m], res[:m], color=color, linewidth=0.7)
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Остаток")
        ax.set_title(f"Остатки: {name}")

    axes[-1].set_xlabel("Дата")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    fig.suptitle("Анализ остатков моделей", fontsize=12)
    fig.tight_layout()
    _save(fig, "07_residuals", show)
