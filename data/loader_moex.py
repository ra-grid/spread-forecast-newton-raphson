"""
Загрузка данных Московской биржи (MOEX ISS).

Источник: открытое HTTP-API биржи https://iss.moex.com/iss/reference/
Библиотека: apimoex (pip install apimoex)

Спред в работе определяется как ДНЕВНОЙ ЛОГАРИФМИЧЕСКИЙ ДОХОД
выбранной валютной пары:
        spread_t = ln(Close_t / Close_{t-1})

Экзогенная переменная — индекс RVI (Russian Volatility Index),
прямой аналог VIX, рассчитываемый MOEX по опционам на фьючерс на индекс РТС.

Контракт функции `load_data` совпадает с data/loader.py:
возвращает DataFrame c колонками ['spread', 'vix'] (поле 'vix'
содержит RVI — имя сохранено для совместимости с остальным кодом).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MOEX_CURRENCY,
    MOEX_INDEX,
    MOEX_CACHE_FILE,
    MOEX_DATA_START,
    MOEX_DATA_END,
)


# ──────────────────────────────────────────────────────────────
#  Низкоуровневые запросы к MOEX ISS через apimoex
# ──────────────────────────────────────────────────────────────

def _download_currency_ohlc(security: str, start: str, end: str) -> pd.DataFrame:
    """
    Дневная история торгов валютной парой на MOEX.

    engine='currency', market='selt' (Селективный валютный рынок),
    board='CETS' (биржевые торги Т+0).
    """
    import apimoex
    import requests

    with requests.Session() as session:
        rows = apimoex.get_board_history(
            session,
            security=security,
            start=start,
            end=end,
            engine="currency",
            market="selt",
            board="CETS",
            columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLRUR"),
        )

    if not rows:
        raise RuntimeError(
            f"MOEX ISS вернул пустой ответ для {security} за {start}—{end}. "
            f"Проверьте тикер и доступность сети."
        )

    df = pd.DataFrame(rows)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.set_index("TRADEDATE").sort_index()
    df = df.rename(columns={
        "OPEN": "open", "HIGH": "high", "LOW": "low",
        "CLOSE": "close", "VOLRUR": "volume_rub",
    })
    df = df[~df.index.duplicated(keep="last")]
    return df


def _download_index(security: str, start: str, end: str) -> pd.Series:
    """
    Дневная история индекса MOEX через get_market_history.

    Этот метод запрашивает все режимы сразу (board=*) и не требует
    знать заранее, в каком режиме торгуется индекс. Подходит для
    RVI, IMOEX, RTSI и т.п. Дубли по дате (если индекс шёл сразу
    в нескольких режимах) дедуплицируются — берём первое значение.
    """
    import apimoex
    import requests

    with requests.Session() as session:
        rows = apimoex.get_market_history(
            session,
            security=security,
            start=start,
            end=end,
            engine="stock",
            market="index",
            columns=("BOARDID", "TRADEDATE", "CLOSE"),
        )

    if not rows:
        # Диагностика: попробуем найти описание тикера, чтобы понять,
        # есть ли он вообще на MOEX и под каким engine/market.
        with requests.Session() as session:
            desc = apimoex.find_security_description(session, security)
        hint = ""
        if desc:
            kv = {d["name"]: d.get("value") for d in desc}
            hint = (f"\nНайденный тикер на MOEX: name='{kv.get('NAME')}', "
                    f"type='{kv.get('TYPE')}', group='{kv.get('GROUP')}'. "
                    f"Возможно нужен другой engine/market.")
        raise RuntimeError(
            f"MOEX ISS вернул пустой ответ для индекса {security} "
            f"за {start}—{end}.{hint}"
        )

    df = pd.DataFrame(rows)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.dropna(subset=["CLOSE"])
    df = df.sort_values("TRADEDATE")
    df = df.drop_duplicates(subset="TRADEDATE", keep="first")
    s = df.set_index("TRADEDATE")["CLOSE"]
    s.name = security
    return s


# ──────────────────────────────────────────────────────────────
#  Главная функция (та же сигнатура, что у loader.load_data)
# ──────────────────────────────────────────────────────────────

def load_data(use_cache: bool = True,
              currency: Optional[str] = None,
              index: Optional[str] = None,
              cache_file: Optional[Path] = None,
              data_start: Optional[str] = None,
              data_end: Optional[str] = None) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками:
        spread — ln(close_t / close_{t-1}) для выбранной валютной пары
        vix    — закрытие RVI (Russian Volatility Index)
                 [имя 'vix' сохранено для совместимости с остальным кодом
                  проекта; физически это российский «индекс страха»]

    Индекс — торговые дни (пересечение дат пары и RVI).

    Все параметры опциональны — при отсутствии берутся значения из config.
    Это позволяет одной и той же функцией грузить разные пары
    с разными кэшами (см. compare_pairs.py).
    """
    currency = currency or MOEX_CURRENCY
    index = index or MOEX_INDEX
    cache_file = cache_file or MOEX_CACHE_FILE
    data_start = data_start or MOEX_DATA_START
    data_end = data_end or MOEX_DATA_END

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"[moex] Данные из кэша {cache_file.name}: {len(df)} строк "
              f"({df.index[0].date()} — {df.index[-1].date()})")
        return df

    print(f"[moex] Загружаю {currency} с MOEX ({data_start} — {data_end})...")
    ccy = _download_currency_ohlc(currency, data_start, data_end)

    # Фильтр: убираем дни без реальных торгов (close == 0 или NaN).
    # У CNYRUB_TOM до конца 2022 часть дней закрывалась нулём,
    # потому что юань на MOEX тогда был практически неликвиден.
    n_before = len(ccy)
    ccy = ccy.replace(0, np.nan).dropna(subset=["close"])
    ccy = ccy[ccy["close"] > 0]
    n_after = len(ccy)
    if n_after < n_before:
        print(f"[moex] Отфильтровано {n_before - n_after} нелеквидных дней "
              f"(close == 0 или NaN); осталось {n_after}.")

    if n_after < 100:
        raise RuntimeError(
            f"После фильтрации осталось всего {n_after} дней — слишком мало. "
            f"Возможно, выбран период до реального запуска торгов по {currency}. "
            f"Попробуйте увеличить MOEX_DATA_START в config.py."
        )

    print(f"[moex] Загружаю индекс {index} с MOEX...")
    idx = _download_index(index, data_start, data_end)

    # Лог-доход по close валютной пары
    spread = np.log(ccy["close"] / ccy["close"].shift(1))
    spread.name = "spread"
    # На всякий случай: после первого валидного дня shift(1) даст NaN,
    # а если в середине окажется большой пропуск — там тоже будут аномалии.
    spread = spread.replace([np.inf, -np.inf], np.nan)

    # RVI закрытие (сохраняем под именем 'vix' для совместимости)
    rvi = idx.copy()
    rvi.name = "vix"

    df = pd.concat([spread, rvi], axis=1, sort=False).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.index.name = "date"

    # Winsorize 4σ — защита от выбросов в обоих рядах
    for col in df.columns:
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mu - 4 * sigma, mu + 4 * sigma)

    if df.isna().any().any() or len(df) < 50:
        raise RuntimeError(
            f"После всех фильтров осталось {len(df)} строк "
            f"({df.isna().sum().to_dict()} NaN). Что-то не так с данными."
        )

    df.to_csv(cache_file)
    print(f"[moex] Загружено и сохранено в {cache_file.name}: {len(df)} строк "
          f"({df.index[0].date()} — {df.index[-1].date()})")
    return df


# ──────────────────────────────────────────────────────────────
#  Запуск как скрипта — для ручной проверки загрузки данных
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(use_cache=False)
    print()
    print(df.describe())
    print()
    print("Корреляция spread ↔ RVI:", df["spread"].corr(df["vix"]).round(4))
    print("Последние 5 строк:")
    print(df.tail())
