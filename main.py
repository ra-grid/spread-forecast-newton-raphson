"""
Главный пайплайн проекта.

Запуск:
    python main.py              # полный прогон
    python main.py --no-backtest # без walk-forward бэктеста (быстро)
    python main.py --show        # показывать графики интерактивно
"""

import argparse
import numpy as np
import pandas as pd

from config import TEST_SIZE, FORECAST_STEPS, LOOK_BACK, RANDOM_SEED

from data import load_data   # выбирает источник по config.DATA_SOURCE
from models.newton_raphson import (
    estimate_ar1_newton, forecast_ar1, fitted_values_ar1,
    parameter_inference_ar1, forecast_ar1_with_ci,
)
from models.arx_model import (
    estimate_arx_newton, forecast_arx, fitted_values_arx,
    parameter_inference_arx, forecast_arx_with_ci,
)
from models.arima_model import fit_arima, forecast_arima, fit_sarimax, forecast_sarimax
from models.lstm_model import UnivariateLSTM, MultivariateLSTM, TENSORFLOW_AVAILABLE
from evaluation.metrics import compute_all, print_metrics
from evaluation.backtest import walk_forward_backtest, summarize_backtest
from visualization.plots import (
    plot_spread_and_vix,
    plot_spread_vix_correlation,
    plot_forecasts,
    plot_backtest_metrics,
    plot_lstm_loss,
    plot_nr_convergence,
    plot_residuals,
    show_all,
)

np.random.seed(RANDOM_SEED)


def _print_inference_table(name: str, inference: dict) -> None:
    """
    Печатает таблицу с оценками параметров: estimate, SE, t-stat, 95%-CI, p-value.
    Звёздочки в столбце 'sig' — общепринятые уровни значимости:
        *** — p<0.001,  ** — p<0.01,  * — p<0.05,  . — p<0.1
    """
    if not inference:
        print(f"  [{name}] инференс недоступен (не сошлось).")
        return

    print(f"\n  {name} — статистика параметров:")
    print(f"  {'param':>6} {'estimate':>12} {'std.err':>12} "
          f"{'t':>8} {'p-value':>10} {'95% CI':>30} sig")
    for p, d in inference.items():
        ci = f"[{d['ci_low']:+.4g}, {d['ci_high']:+.4g}]"
        pv = d["p_value"]
        if   pv < 0.001: star = "***"
        elif pv < 0.01:  star = "** "
        elif pv < 0.05:  star = "*  "
        elif pv < 0.10:  star = ".  "
        else:            star = "   "
        print(f"  {p:>6} {d['estimate']:+12.6g} {d['se']:12.6g} "
              f"{d['t']:8.3f} {pv:10.4f} {ci:>30} {star}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-backtest", action="store_true",
                   help="Пропустить walk-forward бэктест")
    p.add_argument("--show", action="store_true",
                   help="Показывать графики интерактивно")
    p.add_argument("--no-cache", action="store_true",
                   help="Принудительно перезагрузить данные")
    p.add_argument("--no-lstm", action="store_true",
                   help="Пропустить LSTM-модели (быстро, не требует TensorFlow)")
    return p.parse_args()


def run(show: bool = False, run_backtest: bool = True,
        use_cache: bool = True, run_lstm: bool = True):
    # LSTM пропускается если: явно отключён или TF не установлен
    use_lstm = run_lstm and TENSORFLOW_AVAILABLE
    if run_lstm and not TENSORFLOW_AVAILABLE:
        print("[main] TensorFlow не найден — LSTM пропускаем. "
              "Для включения: pip install tensorflow")
    # ── 1. Данные ──────────────────────────────────────────
    print("\n=== Загрузка данных ===")
    df = load_data(use_cache=use_cache)
    spread_all = df["spread"].values
    vix_all = df["vix"].values
    dates_all = df.index

    plot_spread_and_vix(df, show=show)
    plot_spread_vix_correlation(df, show=show)

    # Разбивка train / test
    n = len(spread_all)
    train_end = n - TEST_SIZE

    spread_train = spread_all[:train_end]
    vix_train = vix_all[:train_end]
    spread_test = spread_all[train_end:]
    vix_test = vix_all[train_end:]
    dates_train = dates_all[:train_end]
    dates_test = dates_all[train_end:]

    print(f"Train: {len(spread_train)} дней | Test: {len(spread_test)} дней")

    # ── 2. AR(1) + Newton-Raphson ──────────────────────────
    print("\n=== AR(1) — метод Ньютона-Рафсона ===")
    nr_result = estimate_ar1_newton(spread_train)            # train — для прогноза/диагностики
    nr_full = estimate_ar1_newton(spread_all)                # полная выборка — для инференса (как в Таблице 10)
    print(f"  c={nr_full.c:.6f}  φ={nr_full.phi:.6f}  σ={nr_full.sigma:.6f}")
    print(f"  Итераций: {nr_full.n_iter}  Сошлось: {nr_full.converged}")

    ar1_inference = parameter_inference_ar1(nr_full, spread_all)
    _print_inference_table("AR(1)", ar1_inference)

    # Прогноз AR(1) с 95%-доверительной полосой
    ar1_mean, ar1_lo, ar1_hi = forecast_ar1_with_ci(
        nr_result, spread_train[-1], FORECAST_STEPS
    )
    ar1_fc = ar1_mean
    ar1_fit = fitted_values_ar1(nr_result, spread_train)

    # ── 3. ARX(1) с экзогенной переменной ───────────────────
    print("\n=== ARX(1) с экзогенной переменной ===")
    arx_result = estimate_arx_newton(spread_train, vix_train)     # train — для прогноза/диагностики
    arx_full = estimate_arx_newton(spread_all, vix_all)          # полная выборка — для инференса
    print(f"  c={arx_full.c:.6f}  φ={arx_full.phi:.6f}  "
          f"β={arx_full.beta:.6f}  σ={arx_full.sigma:.6f}")

    arx_inference = parameter_inference_arx(arx_full, spread_all, vix_all)
    _print_inference_table("ARX(1)", arx_inference)

    vix_fc_naive = np.full(FORECAST_STEPS, vix_train[-1])
    arx_mean, arx_lo, arx_hi = forecast_arx_with_ci(
        arx_result, spread_train[-1], vix_fc_naive
    )
    arx_fc = arx_mean
    arx_fit = fitted_values_arx(arx_result, spread_train, vix_train)

    # ── 4. ARIMA ────────────────────────────────────────────
    print("\n=== ARIMA ===")
    arima_fitted = fit_arima(spread_train)
    arima_fc = forecast_arima(arima_fitted, FORECAST_STEPS)
    arima_resid = np.asarray(arima_fitted.resid)

    # ── 5. SARIMAX с VIX ────────────────────────────────────
    print("\n=== SARIMAX + VIX ===")
    sarimax_fitted = fit_sarimax(spread_train, vix_train)
    sarimax_fc = forecast_sarimax(sarimax_fitted, vix_fc_naive, FORECAST_STEPS)
    sarimax_resid = np.asarray(sarimax_fitted.resid)

    # ── 6-7. LSTM (опционально, если TF установлен) ─────────
    lstm_uni_fc = None
    lstm_multi_fc = None
    if use_lstm:
        print("\n=== LSTM (univariate) ===")
        lstm_uni = UnivariateLSTM()
        lstm_uni.fit(spread_train)
        lstm_uni_fc = lstm_uni.forecast(spread_train, FORECAST_STEPS)
        plot_lstm_loss(lstm_uni.history, "LSTM-uni", show=show)

        print("\n=== LSTM (multivariate: спред + экзогенная) ===")
        lstm_multi = MultivariateLSTM()
        lstm_multi.fit(spread_train, vix_train)
        lstm_multi_fc = lstm_multi.forecast(spread_train, vix_train, FORECAST_STEPS)
        plot_lstm_loss(lstm_multi.history, "LSTM-multi", show=show)

    # ── 8. Метрики на тесте ─────────────────────────────────
    print("\n=== Метрики на тесте ===")
    y_true = spread_test[:FORECAST_STEPS]
    forecasts = {
        "AR1":        ar1_fc,
        "ARX":        arx_fc,
        "ARIMA":      arima_fc,
        "SARIMAX":    sarimax_fc,
    }
    if lstm_uni_fc is not None:
        forecasts["LSTM-uni"] = lstm_uni_fc
    if lstm_multi_fc is not None:
        forecasts["LSTM-multi"] = lstm_multi_fc
    for name, fc in forecasts.items():
        m = compute_all(
            y_true,
            np.asarray(fc).flatten()[:len(y_true)],
            y_train=spread_train,
        )
        print_metrics(name, m)

    # ── 9. Визуализация прогнозов ───────────────────────────
    from config import DATA_SOURCE
    if DATA_SOURCE == "moex":
        from config import MOEX_CURRENCY
        asset_for_title = {
            "USD000UTSTOM": "USD/RUB",
            "EUR_RUB__TOM": "EUR/RUB",
            "CNYRUB_TOM":   "CNY/RUB",
        }.get(MOEX_CURRENCY, MOEX_CURRENCY)
    else:
        asset_for_title = "EUR/USD"
    fc_title = (
        f"Прогноз {asset_for_title} на {FORECAST_STEPS} дн.: "
        + ", ".join(forecasts.keys())
    )
    forecasts_ci = {
        "AR1": (ar1_lo, ar1_hi),
        "ARX": (arx_lo, arx_hi),
    }
    plot_forecasts(
        dates_train, spread_train,
        dates_test, spread_test,
        forecasts,
        forecasts_ci=forecasts_ci,
        show=show,
        title=fc_title,
    )

    # Остатки
    res_len = len(arima_resid)
    plot_residuals(
        dates_train[1:res_len + 1],
        {"AR1":   (spread_train[1:] - ar1_fit)[:res_len],
         "ARX":   (spread_train[1:] - arx_fit)[:res_len],
         "ARIMA": arima_resid,
         "SARIMAX": sarimax_resid[:res_len]},
        show=show,
    )

    # ── 10. Walk-forward бэктест ────────────────────────────
    if run_backtest:
        print("\n=== Walk-forward бэктест ===")

        def make_ar1(s, v, steps):
            r = estimate_ar1_newton(s)
            return forecast_ar1(r, s[-1], steps)

        def make_arx(s, v, steps):
            r = estimate_arx_newton(s, v)
            return forecast_arx(r, s[-1], np.full(steps, v[-1]))

        def make_arima(s, v, steps):
            f = fit_arima(s)
            return forecast_arima(f, steps)

        def make_sarimax(s, v, steps):
            f = fit_sarimax(s, v)
            return forecast_sarimax(f, np.full(steps, v[-1]), steps)

        model_fns = {
            "AR1":        make_ar1,
            "ARX":        make_arx,
            "ARIMA":      make_arima,
            "SARIMAX":    make_sarimax,
        }

        if use_lstm:
            def make_lstm_uni(s, v, steps):
                m = UnivariateLSTM()
                m.fit(s)
                return m.forecast(s, steps)

            def make_lstm_multi(s, v, steps):
                m = MultivariateLSTM()
                m.fit(s, v)
                return m.forecast(s, v, steps)

            model_fns["LSTM-uni"] = make_lstm_uni
            model_fns["LSTM-multi"] = make_lstm_multi

        bt_df = walk_forward_backtest(spread_all, vix_all, model_fns, verbose=True)
        summary = summarize_backtest(bt_df)

        print("\n=== Итоговая таблица (walk-forward) ===")
        print(summary.to_string())
        summary.to_csv("results/backtest_summary.csv")

        plot_backtest_metrics(summary, show=show)

    print("\n=== Готово. Результаты сохранены в results/ ===")

    # Если включён --show, открываем все окна разом (блокирующий вызов).
    # Пользователь может переключаться между фигурами, зумить, сохранять.
    if show:
        show_all()


if __name__ == "__main__":
    args = _parse_args()
    run(
        show=args.show,
        run_backtest=not args.no_backtest,
        use_cache=not args.no_cache,
        run_lstm=not args.no_lstm,
    )
