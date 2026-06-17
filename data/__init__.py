"""
Единая точка входа для загрузки данных.

В зависимости от config.DATA_SOURCE подставляется один из loader-ов:
    'yahoo' → data.loader.load_data    (EUR/USD + VIX)
    'moex'  → data.loader_moex.load_data (USD/RUB или CNY/RUB + RVI)

В обоих случаях возвращаемый DataFrame имеет колонки ['spread', 'vix'],
поэтому остальной код (модели, бэктест, визуализация) не меняется.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_SOURCE


if DATA_SOURCE == "moex":
    from data.loader_moex import load_data           # noqa: F401
elif DATA_SOURCE == "yahoo":
    from data.loader import load_data                # noqa: F401
else:
    raise ValueError(
        f"Неизвестное значение DATA_SOURCE='{DATA_SOURCE}'. "
        "Допустимые: 'yahoo', 'moex'."
    )

__all__ = ["load_data"]
