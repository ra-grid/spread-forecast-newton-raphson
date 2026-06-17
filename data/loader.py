"""
Загрузка и подготовка данных EUR/USD и VIX.

Источник: Yahoo Finance через yfinance.

Спред в работе определяется как ДНЕВНОЙ ЛОГАРИФМИЧЕСКИЙ ДОХОД EUR/USD:
        spread_t = ln(Close_t / Close_{t-1})
Это стандартное в финансовой эконометрике определение «дневного спреда
доходностей»: ряд стационарный, имеет среднее ≈ 0, симметричный по знаку
и хорошо подходит для регрессионных моделей AR / ARX.

Замечание: ранее в проекте использовалось определение spread = Close − Open.
Для тикера EUR/USD=X Yahoo Finance с 2022 года возвращает Open == Close для
большинства дней (FX-данные у Yahoo «слипшиеся»), из-за чего ряд вырождался
в константу 0 и сравнение моделей теряло смысл.

VIX — индекс волатильности CBOE (индекс страха), экзогенная переменная Z_t.
"""

import numpy as np
import pandas as pd
import yfinance as yf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EURUSD_TICKER, VIX_TICKER,
    DATA_START, DATA_END,
    YAHOO_CACHE_FILE,
)


def _download_series(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    return df


def load_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками:
        spread  — ln(Close_t / Close_{t-1}) для EUR/USD (дневной лог-доход)
        vix     — VIX закрытие
    Индекс — торговые дни (пересечение дат обоих инструментов).

    Этот loader всегда читает/пишет в YAHOO_CACHE_FILE
    (data/eurusd_vix.csv), независимо от DATA_SOURCE — чтобы
    EUR/USD-данные не путались с MOEX-кэшем в сравнительных скриптах.
    """
    if use_cache and YAHOO_CACHE_FILE.exists():
        df = pd.read_csv(YAHOO_CACHE_FILE, index_col=0, parse_dates=True)
        print(f"[loader] Данные из кэша {YAHOO_CACHE_FILE.name}: "
              f"{len(df)} строк ({df.index[0].date()} — {df.index[-1].date()})")
        return df

    print("[loader] Загружаю EUR/USD...")
    eurusd = _download_series(EURUSD_TICKER, DATA_START, DATA_END)
    print("[loader] Загружаю VIX...")
    vix_raw = _download_series(VIX_TICKER, DATA_START, DATA_END)

    # Лог-доход EUR/USD: spread_t = ln(Close_t / Close_{t-1})
    close = eurusd["Close"].squeeze().astype(float)
    spread = np.log(close / close.shift(1))
    spread.name = "spread"

    # VIX закрытие
    vix = vix_raw["Close"].squeeze().astype(float)
    vix.name = "vix"

    # Объединяем по внутреннему пересечению дат
    df = pd.concat([spread, vix], axis=1).dropna()
    df.index.name = "date"

    # Убираем выбросы: значения дальше 4σ → границу (winsorize)
    for col in df.columns:
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mu - 4 * sigma, mu + 4 * sigma)

    df.to_csv(YAHOO_CACHE_FILE)
    print(f"[loader] Загружено и сохранено: {len(df)} строк "
          f"({df.index[0].date()} — {df.index[-1].date()})")
    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Возвращает:
        X  — DataFrame фич (vix, лаги спреда, скользящие стат.)
        y  — Series целевой переменной (spread)
    """
    d = df.copy()

    # Лаги спреда
    for lag in [1, 2, 3, 5]:
        d[f"spread_lag{lag}"] = d["spread"].shift(lag)

    # Скользящее среднее и стандартное отклонение VIX
    d["vix_ma5"] = d["vix"].rolling(5).mean()
    d["vix_std5"] = d["vix"].rolling(5).std()

    # Скользящая волатильность спреда
    d["spread_vol5"] = d["spread"].rolling(5).std()

    d.dropna(inplace=True)

    feature_cols = [c for c in d.columns if c != "spread"]
    return d[feature_cols], d["spread"]


if __name__ == "__main__":
    df = load_data(use_cache=False)
    print(df.describe())
    print(df.tail())
