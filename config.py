"""
Центральный конфигурационный файл проекта.
Все гиперпараметры, пути и настройки — только здесь.
"""

from datetime import date, timedelta
from pathlib import Path

# Сегодняшняя дата — используется как верхняя граница загрузки данных,
# чтобы кэши всегда тянулись «по сегодня» без ручного обновления.
_TODAY = date.today()
_TOMORROW = (_TODAY + timedelta(days=1)).isoformat()

# ──────────────────────────────────────────────
#  Пути
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
#  Выбор источника данных
# ──────────────────────────────────────────────
# 'yahoo' — EUR/USD + VIX через yfinance
# 'moex'  — USD/RUB или CNY/RUB + RVI через MOEX ISS (apimoex)
DATA_SOURCE = "moex"   # ← переключатель источника

# ──────────────────────────────────────────────
#  Источник 1: Yahoo Finance (EUR/USD + VIX)
# ──────────────────────────────────────────────
EURUSD_TICKER = "EURUSD=X"
VIX_TICKER = "^VIX"
YAHOO_CACHE_FILE = DATA_DIR / "eurusd_vix.csv"
DATA_START = "2018-01-01"
DATA_END = _TOMORROW   # «по сегодня» (yfinance: верхняя граница исключающая)

# ──────────────────────────────────────────────
#  Источник 2: MOEX ISS (USD/RUB или CNY/RUB + RVI)
# ──────────────────────────────────────────────
# Доступные валютные тикеры на MOEX (engine=currency, market=selt, board=CETS):
#   "USD000UTSTOM"  — USD/RUB TOM   (торги приостановлены 13.06.2024, история до этой даты)
#   "EUR_RUB__TOM"  — EUR/RUB TOM   (также приостановлен с 13.06.2024)
#   "CNYRUB_TOM"    — CNY/RUB TOM   (актуальный, торгуется и сейчас)
MOEX_CURRENCY = "CNYRUB_TOM"
MOEX_INDEX = "RVI"          # Russian Volatility Index — аналог VIX
MOEX_CACHE_FILE = DATA_DIR / "moex_rvi.csv"
MOEX_DATA_START = "2018-01-01"
MOEX_DATA_END = _TOMORROW   # «по сегодня»

# Активный кэш — выбирается автоматически в зависимости от DATA_SOURCE
CACHE_FILE = MOEX_CACHE_FILE if DATA_SOURCE == "moex" else YAHOO_CACHE_FILE

# ──────────────────────────────────────────────
#  Разбивка train / test
# ──────────────────────────────────────────────
TEST_SIZE = 30          # дней для теста (walk-forward backtest)
FORECAST_STEPS = 10     # горизонт прогноза (торговых дней)

# ──────────────────────────────────────────────
#  Newton-Raphson / AR(1)
# ──────────────────────────────────────────────
NR_MAX_ITER = 200
NR_TOL = 1e-8

# ──────────────────────────────────────────────
#  ARIMA / SARIMAX
# ──────────────────────────────────────────────
ARIMA_ORDER = (2, 0, 2)
SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL = (1, 0, 1, 5)   # недельная сезонность торговых дней

# ──────────────────────────────────────────────
#  LSTM
# ──────────────────────────────────────────────
LOOK_BACK = 30          # длина окна (дней)
LSTM_UNITS = 64         # нейронов в LSTM-слое
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 150
LSTM_BATCH = 16
LSTM_PATIENCE = 15      # EarlyStopping
LSTM_VAL_SPLIT = 0.1

# ──────────────────────────────────────────────
#  Случайное зерно (воспроизводимость)
# ──────────────────────────────────────────────
RANDOM_SEED = 42
