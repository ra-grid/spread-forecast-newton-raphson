"""
Сравнительный анализ валютных пар.

Идея.  Прогоняет один и тот же конвейер (AR(1)-NR и ARX(1)-NR с
информационной матрицей Фишера) для нескольких валютных пар и сводит
результаты в общую таблицу. Главный объект интереса — коэффициент β
при экзогенной переменной (индекс страха: VIX для EUR/USD, RVI для
рублёвых пар) и его статистическая значимость.

Запуск:
    python compare_pairs.py             # полный прогон с загрузкой данных
    python compare_pairs.py --no-cache  # принудительная перезагрузка

Выход:
    - таблица в консоли
    - results/compare_pairs_summary.csv
    - results/08_beta_forest_plot.png  (forest-plot β c 95%-CI)
    - results/09_compare_spreads.png   (нормированные ряды спредов)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Сделаем модули проекта импортируемыми (config, data.*, models.*)
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, RESULTS_DIR, TEST_SIZE, FORECAST_STEPS,
    YAHOO_CACHE_FILE,
)
from data.loader import load_data as load_yahoo
from data.loader_moex import load_data as load_moex
from models.newton_raphson import (
    estimate_ar1_newton, parameter_inference_ar1, forecast_ar1,
)
from models.arx_model import (
    estimate_arx_newton, parameter_inference_arx, forecast_arx,
)
from evaluation.metrics import compute_all


# ──────────────────────────────────────────────────────────────
#  Конфигурация пар для сравнения
# ──────────────────────────────────────────────────────────────

@dataclass
class PairConfig:
    """Описание одной валютной пары для сравнения."""
    name: str                           # человекочитаемое имя, e.g. "USD/RUB"
    exog_name: str                      # имя экзогенной переменной (VIX или RVI)
    loader: Callable                    # функция загрузки
    loader_kwargs: dict                 # параметры для loader-а


# Готовый список пар. Если нужно — поменяйте/расширьте.
PAIRS: list[PairConfig] = [
    PairConfig(
        name="EUR/USD",
        exog_name="VIX",
        loader=load_yahoo,
        loader_kwargs=dict(),   # Yahoo-loader сам найдёт свой кэш
    ),
    PairConfig(
        name="USD/RUB",
        exog_name="RVI",
        loader=load_moex,
        loader_kwargs=dict(
            currency="USD000UTSTOM",
            data_end="2024-06-13",
            cache_file=DATA_DIR / "usdrub_rvi.csv",
        ),
    ),
    PairConfig(
        name="EUR/RUB",
        exog_name="RVI",
        loader=load_moex,
        loader_kwargs=dict(
            currency="EUR_RUB__TOM",
            data_end="2024-06-13",
            cache_file=DATA_DIR / "eurrub_rvi.csv",
        ),
    ),
    PairConfig(
        name="CNY/RUB",
        exog_name="RVI",
        loader=load_moex,
        loader_kwargs=dict(
            currency="CNYRUB_TOM",
            cache_file=DATA_DIR / "moex_rvi.csv",  # уже скачан
        ),
    ),
]


# ──────────────────────────────────────────────────────────────
#  Пайплайн для одной пары
# ──────────────────────────────────────────────────────────────

def run_one_pair(pair: PairConfig, use_cache: bool = True) -> dict:
    """
    Один прогон: AR(1)-NR + ARX(1)-NR + инференс + метрики на тесте.
    Возвращает словарь с результатами для сводной таблицы.
    """
    print(f"\n{'=' * 60}")
    print(f"  Пара: {pair.name}  (экзогенная: {pair.exog_name})")
    print('=' * 60)

    df = pair.loader(use_cache=use_cache, **pair.loader_kwargs)
    spread_all = df["spread"].values.astype(float)
    exog_all = df["vix"].values.astype(float)

    n = len(spread_all)
    train_end = n - TEST_SIZE
    spread_train = spread_all[:train_end]
    exog_train = exog_all[:train_end]
    spread_test = spread_all[train_end:]
    exog_test = exog_all[train_end:]

    # AR(1) и ARX(1) — обучение на train ТОЛЬКО для прогноза/MASE (без утечки теста)
    ar_tr = estimate_ar1_newton(spread_train)
    ar_fc = forecast_ar1(ar_tr, spread_train[-1], FORECAST_STEPS)
    arx_tr = estimate_arx_newton(spread_train, exog_train)
    arx_fc = forecast_arx(arx_tr, spread_train[-1],
                          np.full(FORECAST_STEPS, exog_train[-1]))

    # Оценка параметров и тест значимости β — на ПОЛНОЙ выборке (как в Таблице 10 ВКР)
    ar = estimate_ar1_newton(spread_all)
    ar_inf = parameter_inference_ar1(ar, spread_all)
    arx = estimate_arx_newton(spread_all, exog_all)
    arx_inf = parameter_inference_arx(arx, spread_all, exog_all)

    # Корреляция спреда с экзогенной (на всей выборке)
    corr = float(np.corrcoef(spread_all, exog_all)[0, 1])

    # Доля дисперсии, объяснённая включением экзогенной (Δσ²/σ²) — на полной выборке
    sigma_ar = ar.sigma
    sigma_arx = arx.sigma
    r2_exog = 1.0 - (sigma_arx ** 2) / (sigma_ar ** 2) if sigma_ar > 0 else np.nan

    # Метрики на тесте
    y_true = spread_test[:FORECAST_STEPS]
    m_ar = compute_all(y_true, np.asarray(ar_fc).flatten()[:len(y_true)],
                       y_train=spread_train)
    m_arx = compute_all(y_true, np.asarray(arx_fc).flatten()[:len(y_true)],
                        y_train=spread_train)

    # Достаём beta-инференс
    beta = arx_inf.get("beta", {})
    phi = arx_inf.get("phi", {})

    print(f"  n={n}, train={train_end}, test={TEST_SIZE}")
    print(f"  AR(1):  c={ar.c:+.4e}  φ={ar.phi:+.5f}  σ={ar.sigma:.6f}")
    print(f"  ARX(1): c={arx.c:+.4e}  φ={arx.phi:+.5f}  "
          f"β={arx.beta:+.4e}  σ={arx.sigma:.6f}")
    if beta:
        print(f"  β-инференс: t={beta['t']:+.3f}  p={beta['p_value']:.4f}  "
              f"95%-CI=[{beta['ci_low']:+.3e}, {beta['ci_high']:+.3e}]")
    print(f"  Корреляция спред↔экзогенная: r = {corr:+.4f}")
    print(f"  Доля дисперсии, объяснённая экзогенной: R²_exog = {100 * r2_exog:.3f}%")
    print(f"  Тест MASE: AR={m_ar['MASE']:.4f}   ARX={m_arx['MASE']:.4f}")

    return {
        "pair": pair.name,
        "exog": pair.exog_name,
        "n": n,
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "corr": corr,
        "ar_phi": ar.phi,
        "ar_sigma": ar.sigma,
        "arx_c": arx.c,
        "arx_phi": arx.phi,
        "arx_beta": arx.beta,
        "arx_sigma": arx.sigma,
        "beta_se":      beta.get("se", np.nan),
        "beta_t":       beta.get("t", np.nan),
        "beta_p":       beta.get("p_value", np.nan),
        "beta_ci_low":  beta.get("ci_low", np.nan),
        "beta_ci_high": beta.get("ci_high", np.nan),
        "phi_p":        phi.get("p_value", np.nan),
        "r2_exog":      r2_exog,
        "mase_ar":      m_ar["MASE"],
        "mase_arx":     m_arx["MASE"],
        "theilu_ar":    m_ar["TheilU"],
        "theilu_arx":   m_arx["TheilU"],
    }


# ──────────────────────────────────────────────────────────────
#  Сводная таблица и визуализация
# ──────────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""


def print_summary(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Пара":      r["pair"],
            "Экзог.":    r["exog"],
            "n":         r["n"],
            "Период":    f"{r['start']} — {r['end']}",
            "Corr":      f"{r['corr']:+.4f}",
            "β":         f"{r['arx_beta']:+.3e}",
            "t":         f"{r['beta_t']:+.2f}",
            "p":         f"{r['beta_p']:.4f}",
            "sig":       _sig_stars(r["beta_p"]),
            "95%-CI":    f"[{r['beta_ci_low']:+.2e}, {r['beta_ci_high']:+.2e}]",
            "R²_exog,%": f"{100 * r['r2_exog']:.3f}",
            "MASE_AR":   f"{r['mase_ar']:.4f}",
            "MASE_ARX":  f"{r['mase_arx']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("  СВОДКА: коэффициент β у ARX(1) для разных валютных пар")
    print("=" * 80)
    # Печать без обрезки
    with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.max_colwidth", 40):
        print(df.to_string(index=False))
    return df


def plot_beta_forest(results: list[dict], show: bool = False) -> None:
    """
    Forest-plot β-коэффициентов: точка + 95%-CI горизонтальной линией,
    разные пары на разных строках. Классическая визуализация
    мета-анализа, идеально подходит для сравнительной главы.
    """
    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.8 * len(results)))

    ys = np.arange(len(results))[::-1]
    for y, r in zip(ys, results):
        b = r["arx_beta"]
        lo = r["beta_ci_low"]
        hi = r["beta_ci_high"]
        color = "tab:red" if r["beta_p"] < 0.05 else "tab:gray"
        ax.errorbar(b, y, xerr=[[b - lo], [hi - b]],
                    fmt="o", color=color, ecolor=color,
                    elinewidth=2, capsize=5, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.7)
        sig = _sig_stars(r["beta_p"])
        ax.text(hi, y, f"  {sig}  p={r['beta_p']:.3f}",
                va="center", fontsize=9)

    ax.axvline(0, linestyle="--", color="black", linewidth=1, alpha=0.6,
               label="β = 0 (нет эффекта)")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['pair']} ({r['exog']})" for r in results])
    ax.set_xlabel("Коэффициент β при экзогенной переменной (точка ± 95%-CI)")
    ax.set_title("Сравнение силы влияния индекса страха на спред разных валютных пар\n"
                 "Красным — статистически значимые (p < 0.05), серым — нет")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = RESULTS_DIR / "08_beta_forest_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Сохранено: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_spreads_comparison(dfs: dict[str, pd.DataFrame], show: bool = False) -> None:
    """
    Нормированные спреды всех пар на одних осях (для визуального сравнения
    периодов стресса и сравнимости волатильностей).
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    for name, df in dfs.items():
        s = df["spread"]
        # Нормируем: делим на std (z-score без вычитания среднего,
        # т.к. для лог-доходов среднее ≈ 0)
        z = s / s.std()
        ax.plot(df.index, z, alpha=0.6, linewidth=0.7, label=name)
    ax.set_title("Нормированные спреды (лог-доходы / σ) разных валютных пар")
    ax.set_ylabel("Стандартизированный лог-доход")
    ax.set_xlabel("Дата")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / "09_compare_spreads.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Сохранено: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  Главная функция
# ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true",
                   help="Принудительно перезагрузить данные с биржи")
    p.add_argument("--show", action="store_true",
                   help="Показывать графики интерактивно")
    args = p.parse_args()

    results = []
    dfs_for_plot = {}
    for pair in PAIRS:
        try:
            r = run_one_pair(pair, use_cache=not args.no_cache)
            results.append(r)
            # Перезагрузка для plot_spreads_comparison (всегда из кэша)
            dfs_for_plot[pair.name] = pair.loader(use_cache=True, **pair.loader_kwargs)
        except Exception as e:
            print(f"\n[!] Ошибка при обработке {pair.name}: {e}")
            print(f"    Пара пропущена. Продолжаю с остальными.")
            continue

    if not results:
        print("\nНи одна пара не была обработана. Проверьте подключение к MOEX/Yahoo.")
        return

    # Сводка и графики
    summary_df = print_summary(results)
    summary_df.to_csv(RESULTS_DIR / "compare_pairs_summary.csv", index=False)
    print(f"\n[save] {RESULTS_DIR / 'compare_pairs_summary.csv'}")

    plot_beta_forest(results, show=args.show)
    plot_spreads_comparison(dfs_for_plot, show=args.show)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
