"""
Streamlit-приложение для дипломной работы.

Запуск:
    streamlit run app.py
        ↑ откроется в браузере по адресу http://localhost:8501

Вкладки:
  📊 Данные      — обзор, корреляция, базовая статистика
  🔬 Модели      — AR(1)-NR и ARX(1)-NR + статистический инференс
  🔮 Прогноз     — точечный прогноз с 95%-доверительной полосой, метрики
  📈 Сравнение   — forest-plot β по четырём валютным парам

Архитектура: интерфейс — это обёртка над уже написанными модулями
проекта (data/, models/, evaluation/, visualization/), а не дублирование
их функциональности.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Подтягиваем модули проекта
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, RESULTS_DIR, NR_MAX_ITER, NR_TOL,
)
from data.loader import load_data as load_yahoo
from data.loader_moex import load_data as load_moex
from models.newton_raphson import (
    estimate_ar1_newton, forecast_ar1_with_ci,
    parameter_inference_ar1, fitted_values_ar1,
)
from models.arx_model import (
    estimate_arx_newton, forecast_arx_with_ci,
    parameter_inference_arx, fitted_values_arx,
)
from evaluation.metrics import compute_all


# ──────────────────────────────────────────────────────────────
#  Конфигурация валютных пар
# ──────────────────────────────────────────────────────────────

@dataclass
class PairSpec:
    name: str
    exog: str
    loader: callable
    loader_kwargs: dict
    description: str


PAIRS = {
    "EUR/USD": PairSpec(
        name="EUR/USD",
        exog="VIX",
        loader=load_yahoo,
        loader_kwargs=dict(),
        description="Контрольная пара (Yahoo). Свободно плавающие нероссийские валюты.",
    ),
    "USD/RUB": PairSpec(
        name="USD/RUB",
        exog="RVI",
        loader=load_moex,
        loader_kwargs=dict(
            currency="USD000UTSTOM",
            data_end="2024-06-13",
            cache_file=DATA_DIR / "usdrub_rvi.csv",
        ),
        description="Биржевые торги USD/RUB на MOEX (приостановлены 13.06.2024 санкциями OFAC).",
    ),
    "EUR/RUB": PairSpec(
        name="EUR/RUB",
        exog="RVI",
        loader=load_moex,
        loader_kwargs=dict(
            currency="EUR_RUB__TOM",
            data_end="2024-06-13",
            cache_file=DATA_DIR / "eurrub_rvi.csv",
        ),
        description="Биржевые торги EUR/RUB на MOEX (приостановлены 13.06.2024).",
    ),
    "CNY/RUB": PairSpec(
        name="CNY/RUB",
        exog="RVI",
        loader=load_moex,
        loader_kwargs=dict(
            currency="CNYRUB_TOM",
            cache_file=DATA_DIR / "moex_rvi.csv",
        ),
        description="Юань/рубль — основная биржевая пара МосБиржи после 13.06.2024.",
    ),
}


# ──────────────────────────────────────────────────────────────
#  Кэширование тяжёлых операций
# ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Загружаю данные...")
def cached_load(pair_name: str) -> pd.DataFrame:
    spec = PAIRS[pair_name]
    return spec.loader(use_cache=True, **spec.loader_kwargs)


@st.cache_data(show_spinner=False)
def cached_fit_ar1(spread_train_tuple):
    spread_train = np.array(spread_train_tuple)
    r = estimate_ar1_newton(spread_train)
    inf = parameter_inference_ar1(r, spread_train)
    return r, inf


@st.cache_data(show_spinner=False)
def cached_fit_arx(spread_train_tuple, vix_train_tuple):
    spread_train = np.array(spread_train_tuple)
    vix_train = np.array(vix_train_tuple)
    r = estimate_arx_newton(spread_train, vix_train)
    inf = parameter_inference_arx(r, spread_train, vix_train)
    return r, inf


# ──────────────────────────────────────────────────────────────
#  Вспомогательные функции UI
# ──────────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""


def inference_to_df(inference: dict) -> pd.DataFrame:
    rows = []
    for name, d in inference.items():
        rows.append({
            "Параметр":  name,
            "Оценка":    f"{d['estimate']:+.4e}",
            "Std. err.": f"{d['se']:.4e}",
            "t":         f"{d['t']:+.3f}",
            "p-value":   f"{d['p_value']:.4f}",
            "95% CI":    f"[{d['ci_low']:+.3e}, {d['ci_high']:+.3e}]",
            "Знач.":     _sig_stars(d["p_value"]),
        })
    return pd.DataFrame(rows)


def color_significance(val: str) -> str:
    """Жёлтая подсветка для p < 0.05, серая для p ≥ 0.05."""
    try:
        p = float(val)
        if p < 0.001: return "background-color: #ff9966"  # явно значим
        if p < 0.01:  return "background-color: #ffcc99"
        if p < 0.05:  return "background-color: #ffe6cc"
        if p < 0.10:  return "background-color: #fff4e6"
    except (ValueError, TypeError):
        pass
    return ""


# ──────────────────────────────────────────────────────────────
#  Главное приложение
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Прогноз спредов методом Ньютона-Рафсона",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Прогнозирование финансовых спредов")
st.caption("Модели AR(1) и ARX(1) методом Ньютона-Рафсона • дипломная работа • "
           "сравнение валютных пар EUR/USD, USD/RUB, EUR/RUB, CNY/RUB")

# ── Сайдбар ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Параметры анализа")

    pair_name = st.selectbox(
        "Валютная пара",
        list(PAIRS.keys()),
        index=3,  # CNY/RUB по умолчанию — у неё значимый β
        help="Выберите пару. Каждая использует свой источник: "
             "EUR/USD → Yahoo+VIX, остальные → MOEX+RVI.",
    )
    spec = PAIRS[pair_name]
    st.caption(spec.description)

    st.markdown("---")

    forecast_steps = st.slider(
        "Горизонт прогноза (торговых дней)",
        min_value=5, max_value=30, value=10, step=1,
        help="На сколько шагов вперёд прогнозируем. "
             "10 дней ≈ 2 рабочие недели, 21 ≈ месяц.",
    )

    test_size = st.slider(
        "Размер тестового окна",
        min_value=10, max_value=60, value=30, step=5,
        help="Сколько последних дней отрезается как hold-out для оценки моделей.",
    )

    st.markdown("---")
    st.caption("**Уровни значимости:** *** p<0.001 • ** p<0.01 • * p<0.05 • . p<0.10")

# ── Загрузка данных ──────────────────────────────────────────
try:
    df = cached_load(pair_name)
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

spread_all = df["spread"].values.astype(float)
exog_all = df["vix"].values.astype(float)
dates_all = df.index

n = len(spread_all)
if test_size >= n:
    st.error(f"Тестовое окно ({test_size}) больше всей выборки ({n}). Уменьшите.")
    st.stop()
train_end = n - test_size
spread_train = spread_all[:train_end]
exog_train = exog_all[:train_end]
spread_test = spread_all[train_end:]
exog_test = exog_all[train_end:]
dates_train = dates_all[:train_end]
dates_test = dates_all[train_end:]


# ── Вкладки ──────────────────────────────────────────────────
tab_data, tab_models, tab_forecast, tab_compare = st.tabs([
    "📊 Данные", "🔬 Модели (NR + инференс)",
    "🔮 Прогноз", "📈 Сравнение пар",
])

# ════════════════════════════════════════════════════════════
#  Вкладка 1 — Данные
# ════════════════════════════════════════════════════════════
with tab_data:
    st.subheader(f"Обзор: {pair_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Наблюдений", f"{n:,}")
    c2.metric("Период", f"{dates_all[0].date()}\n— {dates_all[-1].date()}")
    c3.metric("σ спреда", f"{spread_all.std():.5f}")
    corr = float(np.corrcoef(spread_all, exog_all)[0, 1])
    c4.metric(f"Корр. со {spec.exog}", f"{corr:+.4f}")

    st.caption(
        f"ℹ️ **{spec.exog}** — индекс волатильности рынка акций "
        "(VIX — по опционам на S&P 500, RVI — по опционам на индекс РТС), "
        "а не волатильность самой валютной пары: собственного индекса волатильности "
        "у пары не существует, поэтому используется общерыночный индикатор риск-аппетита "
        "(«индекс страха»)."
    )

    st.markdown("#### Временные ряды")

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                         subplot_titles=(f"Лог-доходность {pair_name}", spec.exog))
    fig1.add_trace(go.Scatter(
        x=dates_all, y=spread_all, mode="lines",
        line=dict(color="#7fb8e0", width=0.8), name=f"Спред {pair_name}",
        hovertemplate="%{x|%Y-%m-%d}<br>спред = %{y:.5f}<extra></extra>"), row=1, col=1)
    fig1.add_hline(y=0, line=dict(color="black", width=0.5), opacity=0.4, row=1, col=1)
    fig1.add_trace(go.Scatter(
        x=dates_all, y=exog_all, mode="lines",
        line=dict(color="#e377c2", width=0.8), name=spec.exog,
        hovertemplate="%{x|%Y-%m-%d}<br>" + spec.exog + " = %{y:.2f}<extra></extra>"), row=2, col=1)
    fig1.update_layout(height=520, showlegend=False, hovermode="x unified",
                       margin=dict(l=50, r=20, t=40, b=40))
    fig1.update_xaxes(title_text="Дата", row=2, col=1)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Корреляция")
    z = np.polyfit(exog_all, spread_all, 1)
    x_line = np.linspace(exog_all.min(), exog_all.max(), 200)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=exog_all, y=spread_all, mode="markers",
        marker=dict(color="#e377c2", size=4, opacity=0.3), name="наблюдения",
        hovertemplate=spec.exog + " = %{x:.2f}<br>спред = %{y:.5f}<extra></extra>"))
    fig2.add_trace(go.Scatter(
        x=x_line, y=np.poly1d(z)(x_line), mode="lines",
        line=dict(color="red", width=2),
        name=f"y = {z[0]:+.5f}x + {z[1]:+.5f}"))
    fig2.add_hline(y=0, line=dict(color="black", width=0.4), opacity=0.4)
    fig2.update_layout(height=430, title=f"r = {corr:+.4f}",
                       xaxis_title=spec.exog, yaxis_title=f"Лог-доход {pair_name}",
                       margin=dict(l=50, r=20, t=50, b=40),
                       legend=dict(orientation="h", y=1.10))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Описательная статистика"):
        st.dataframe(df.describe(), use_container_width=True)


# ════════════════════════════════════════════════════════════
#  Вкладка 2 — Модели + инференс
# ════════════════════════════════════════════════════════════
with tab_models:
    st.subheader("AR(1) и ARX(1) методом Ньютона-Рафсона")

    st.markdown(
        "Параметры оцениваются итеративно: "
        "$\\theta_{k+1} = \\theta_k - H(\\theta_k)^{-1} \\nabla(\\theta_k)$. "
        "Обратная матрица Гессе на сошедшейся точке "
        "**одновременно** даёт ковариационную матрицу оценок параметров — "
        "$\\mathrm{Cov}(\\hat{\\theta}) \\approx H(\\hat{\\theta})^{-1}$. "
        "Из неё получаются стандартные ошибки, t-статистики и доверительные интервалы."
    )

    # Оцениваем модели (с кэшированием)
    spread_t_tuple = tuple(spread_train)
    exog_t_tuple = tuple(exog_train)
    # train-фиты — для прогноза/MASE (вкладка «Прогноз»), без утечки теста
    ar_result, _ = cached_fit_ar1(spread_t_tuple)
    arx_result, _ = cached_fit_arx(spread_t_tuple, exog_t_tuple)
    # полная выборка — для оценок и инференса (как в Таблице 10 диплома)
    ar_full, ar_inf = cached_fit_ar1(tuple(spread_all))
    arx_full, arx_inf = cached_fit_arx(tuple(spread_all), tuple(exog_all))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### AR(1)")
        st.code(
            f"X_t = c + φ·X_{{t-1}} + ε_t\n\n"
            f"c     = {ar_full.c:+.6e}\n"
            f"φ     = {ar_full.phi:+.6f}\n"
            f"σ     = {ar_full.sigma:.6f}\n"
            f"итер. = {ar_full.n_iter}\n"
            f"сошёлся: {ar_full.converged}",
        )
        df_ar = inference_to_df(ar_inf)
        st.dataframe(
            df_ar.style.map(color_significance, subset=["p-value"]),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.markdown(f"#### ARX(1) с экзогенной переменной ({spec.exog})")
        st.code(
            f"X_t = c + φ·X_{{t-1}} + β·Z_t + ε_t\n\n"
            f"c     = {arx_full.c:+.6e}\n"
            f"φ     = {arx_full.phi:+.6f}\n"
            f"β     = {arx_full.beta:+.6e}\n"
            f"σ     = {arx_full.sigma:.6f}\n"
            f"итер. = {arx_full.n_iter}\n"
            f"сошёлся: {arx_full.converged}",
        )
        df_arx = inference_to_df(arx_inf)
        st.dataframe(
            df_arx.style.map(color_significance, subset=["p-value"]),
            use_container_width=True, hide_index=True,
        )

    # Главный вывод про β
    beta = arx_inf.get("beta", {})
    if beta:
        p = beta["p_value"]
        b = beta["estimate"]
        if p < 0.05:
            st.success(
                f"✅ **β статистически значимо отличается от нуля** "
                f"(t = {beta['t']:+.2f}, p = {p:.4f}). "
                f"95%-CI для β: `[{beta['ci_low']:+.3e}, {beta['ci_high']:+.3e}]` — "
                f"{'целиком отрицательный' if beta['ci_high'] < 0 else 'целиком положительный' if beta['ci_low'] > 0 else 'пересекает ноль'}. "
                f"На этой выборке влияние индекса {spec.exog} на спред {pair_name} обнаружено (при сравнении пар — с поправкой на множественные сравнения)."
            )
        elif p < 0.10:
            st.warning(
                f"⚠ β значим только на 10%-уровне "
                f"(t = {beta['t']:+.2f}, p = {p:.4f}). "
                f"Это маржинальная значимость — не отвергаем на стандартном 5%-уровне."
            )
        else:
            st.info(
                f"ℹ️ β не значим (p = {p:.4f}). "
                f"Гипотезу о нулевом влиянии {spec.exog} на {pair_name} нельзя отвергнуть."
            )

    # Доля объяснённой дисперсии
    sigma_ar = ar_full.sigma
    sigma_arx = arx_full.sigma
    r2_exog = 1 - (sigma_arx ** 2) / (sigma_ar ** 2) if sigma_ar > 0 else np.nan

    st.markdown("#### Дополнительные показатели")
    c1, c2, c3 = st.columns(3)
    c1.metric("σ AR(1)", f"{sigma_ar:.6f}")
    c2.metric("σ ARX(1)", f"{sigma_arx:.6f}",
              delta=f"{100*(sigma_arx-sigma_ar)/sigma_ar:+.3f}%",
              delta_color="inverse")
    c3.metric(f"R²_{spec.exog}", f"{100*r2_exog:.3f}%",
              help="Доля дисперсии спреда, объяснённая включением экзогенной переменной")


# ════════════════════════════════════════════════════════════
#  Вкладка 3 — Прогноз с CI
# ════════════════════════════════════════════════════════════
with tab_forecast:
    st.subheader(f"Прогноз на {forecast_steps} торговых дней вперёд")

    ar_mean, ar_lo, ar_hi = forecast_ar1_with_ci(
        ar_result, spread_train[-1], forecast_steps
    )
    vix_future = np.full(forecast_steps, exog_train[-1])
    arx_mean, arx_lo, arx_hi = forecast_arx_with_ci(
        arx_result, spread_train[-1], vix_future
    )

    # Метрики на тесте
    y_true = spread_test[:forecast_steps]
    n_eval = len(y_true)
    metrics = {
        "AR(1)":  compute_all(y_true, ar_mean[:n_eval], y_train=spread_train),
        "ARX(1)": compute_all(y_true, arx_mean[:n_eval], y_train=spread_train),
        "Naive(0)":     compute_all(y_true, np.zeros(n_eval), y_train=spread_train),
        "Naive(last)":  compute_all(y_true, np.full(n_eval, spread_train[-1]),
                                    y_train=spread_train),
    }
    df_metrics = pd.DataFrame(metrics).T[["MAE", "RMSE", "MASE", "TheilU"]]
    df_metrics.index.name = "Модель"

    # График
    ctx = len(dates_test)
    horizon_dates = list(dates_test[:forecast_steps])
    band_x = horizon_dates + horizon_dates[::-1]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(dates_train[-ctx:]), y=list(spread_train[-ctx:]), mode="lines",
        line=dict(color="#aec7e8", width=1.2), name="Train (хвост)"))
    fig3.add_trace(go.Scatter(
        x=list(dates_test), y=list(spread_test), mode="lines",
        line=dict(color="black", width=1.8), name="Факт (тест)"))
    # AR(1) доверительная полоса + прогноз
    fig3.add_trace(go.Scatter(
        x=band_x, y=list(ar_hi) + list(ar_lo)[::-1], fill="toself",
        fillcolor="rgba(31,119,180,0.12)", line=dict(width=0),
        name="AR(1) 95% CI", hoverinfo="skip"))
    fig3.add_trace(go.Scatter(
        x=horizon_dates, y=list(ar_mean), mode="lines+markers",
        line=dict(color="#1f77b4", width=2, dash="dash"), marker=dict(size=5),
        name="AR(1)"))
    # ARX(1) доверительная полоса + прогноз
    fig3.add_trace(go.Scatter(
        x=band_x, y=list(arx_hi) + list(arx_lo)[::-1], fill="toself",
        fillcolor="rgba(255,127,14,0.12)", line=dict(width=0),
        name="ARX(1) 95% CI", hoverinfo="skip"))
    fig3.add_trace(go.Scatter(
        x=horizon_dates, y=list(arx_mean), mode="lines+markers",
        line=dict(color="#ff7f0e", width=2, dash="dash"), marker=dict(size=5),
        name=f"ARX(1)+{spec.exog}"))
    fig3.update_layout(height=540, hovermode="x unified",
                       title=f"{pair_name}: прогноз на {forecast_steps} дн. ± 95%-CI",
                       xaxis_title="Дата", yaxis_title=f"Лог-доход {pair_name}",
                       margin=dict(l=50, r=20, t=50, b=40),
                       legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Метрики на тестовом окне")
    st.caption("**MASE < 1** — модель лучше naive «нет изменения». "
               "**Theil U < 1** — лучше naive «последнее значение».")

    def fmt_metrics(v):
        return f"{v:.5f}"

    styled = df_metrics.style.format(fmt_metrics).background_gradient(
        cmap="RdYlGn_r", subset=["MAE", "RMSE", "MASE"], axis=0)
    st.dataframe(styled, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  Вкладка 4 — Сравнение пар
# ════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Сравнение β по четырём валютным парам")
    st.caption("Один и тот же конвейер ARX(1)-NR применён к каждой паре. "
               "Главный объект сравнения — коэффициент β при индексе страха.")

    summary_path = RESULTS_DIR / "compare_pairs_summary.csv"

    if not summary_path.exists():
        st.warning(
            "Сводный файл `results/compare_pairs_summary.csv` не найден. "
            "Запустите один раз `python compare_pairs.py` в терминале — "
            "она построит сравнительный анализ. "
            "После этого обновите страницу — таблица и forest-plot появятся здесь."
        )
    else:
        cmp_df = pd.read_csv(summary_path)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # forest-plot β (из сводной таблицы) — интерактивный
        st.markdown("#### Forest-plot β (точка ± 95%-CI)")
        fp = cmp_df.reset_index(drop=True)
        se = fp["β"].abs() / fp["t"].abs()
        lo = fp["β"] - 1.96 * se
        hi = fp["β"] + 1.96 * se
        colors = ["#d62728" if pp < 0.05 else "#7f7f7f" for pp in fp["p"]]
        figf = go.Figure()
        for i in range(len(fp)):
            figf.add_trace(go.Scatter(
                x=[lo[i], hi[i]], y=[fp["Пара"][i], fp["Пара"][i]], mode="lines",
                line=dict(color=colors[i], width=3), showlegend=False, hoverinfo="skip"))
        figf.add_trace(go.Scatter(
            x=fp["β"], y=fp["Пара"], mode="markers",
            marker=dict(color=colors, size=13, line=dict(color="black", width=1)),
            showlegend=False,
            customdata=np.stack([fp["t"], fp["p"]], axis=-1),
            hovertemplate="%{y}<br>β = %{x:.2e}<br>t = %{customdata[0]:.2f}, p = %{customdata[1]:.4f}<extra></extra>"))
        figf.add_vline(x=0, line=dict(color="black", dash="dash", width=1))
        figf.update_layout(height=360,
                           xaxis_title="β при индексе страха (красный — p < 0.05)",
                           margin=dict(l=90, r=20, t=20, b=50))
        st.plotly_chart(figf, use_container_width=True)

        # нормированные ряды спредов всех пар — интерактивные
        st.markdown("#### Нормированные ряды спредов всех пар (z-оценка)")
        cache_map = {
            "EUR/USD": DATA_DIR / "eurusd_vix.csv",
            "USD/RUB": DATA_DIR / "usdrub_rvi.csv",
            "EUR/RUB": DATA_DIR / "eurrub_rvi.csv",
            "CNY/RUB": DATA_DIR / "moex_rvi.csv",
        }
        palette = {"EUR/USD": "#1f77b4", "USD/RUB": "#ff7f0e",
                   "EUR/RUB": "#2ca02c", "CNY/RUB": "#d62728"}
        figs = go.Figure()
        for pname, cpath in cache_map.items():
            if cpath.exists():
                dd = pd.read_csv(cpath, parse_dates=["date"])
                s = dd["spread"].values.astype(float)
                sd = s.std()
                sn = (s - s.mean()) / sd if sd > 0 else s
                figs.add_trace(go.Scatter(
                    x=dd["date"], y=sn, mode="lines", name=pname,
                    line=dict(color=palette[pname], width=0.8), opacity=0.85))
        figs.update_layout(height=430, hovermode="x unified",
                           xaxis_title="Дата", yaxis_title="z-норм. спред",
                           margin=dict(l=50, r=20, t=20, b=40),
                           legend=dict(orientation="h", y=1.10))
        st.plotly_chart(figs, use_container_width=True)

        st.info(
            "**Главный эмпирический результат**: единственная пара с β, "
            "формально значимым на 5%-уровне, — **CNY/RUB** (p ≈ 0,02). Эффект, однако, "
            "экономически мал и неустойчив: с поправкой на множественные сравнения "
            "(четыре пары) он незначим, а на под-выборке 2023–2026 годов исчезает и меняет знак. "
            "Значимость на полной выборке порождена турбулентностью 2022 года и структурным "
            "сдвигом 13.06.2024 (остановка торгов USD/RUB и EUR/RUB), а не устойчивой "
            "предсказуемостью пары."
        )


# Подвал
st.markdown("---")
st.caption(
    "Newton-Raphson MLE • Информационная матрица Фишера для инференса • "
    "Walk-forward бэктест в `python main.py` • "
    "Сравнительный анализ в `python compare_pairs.py`"
)
