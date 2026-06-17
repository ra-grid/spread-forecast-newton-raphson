"""
LSTM модель для прогнозирования спреда.

Два варианта:
  1. UnivariateLSTM — только ряд спреда (look_back × 1)
  2. MultivariateLSTM — спред + VIX (look_back × 2)

Оба варианта используют одну и ту же архитектуру:
    Input → LSTM(64, seq) → Dropout → LSTM(64) → Dropout → Dense(1)
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LOOK_BACK,
    LSTM_UNITS,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_BATCH,
    LSTM_PATIENCE,
    LSTM_VAL_SPLIT,
    RANDOM_SEED,
    FORECAST_STEPS,
)

# TensorFlow — опциональная зависимость. Если он не установлен,
# модуль всё равно импортируется, но при попытке использовать LSTM
# будет внятная ошибка с подсказкой как поставить TF.
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    TENSORFLOW_AVAILABLE = False


def _require_tf():
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "Для LSTM-моделей нужен TensorFlow. Установите его командой:\n"
            "    pip install tensorflow\n"
            "Если TF не нужен — запускайте main.py с флагом --no-lstm."
        )


def _set_seed():
    _require_tf()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def _build_model(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOK_BACK, n_features)),
        tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True),
        tf.keras.layers.Dropout(LSTM_DROPOUT),
        tf.keras.layers.LSTM(LSTM_UNITS),
        tf.keras.layers.Dropout(LSTM_DROPOUT),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ──────────────────────────────────────────────
#  Univariate (только спред)
# ──────────────────────────────────────────────

class UnivariateLSTM:
    def __init__(self, look_back: int = LOOK_BACK):
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        self.model: tf.keras.Model | None = None
        self.history = None

    def _make_sequences(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(len(scaled) - self.look_back):
            X.append(scaled[i: i + self.look_back])
            y.append(scaled[i + self.look_back, 0])
        return np.array(X), np.array(y)

    def fit(self, spread: np.ndarray) -> "UnivariateLSTM":
        _set_seed()
        scaled = self.scaler.fit_transform(spread.reshape(-1, 1))
        X, y = self._make_sequences(scaled)
        self.model = _build_model(n_features=1)
        cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=LSTM_PATIENCE,
            restore_best_weights=True, verbose=0,
        )
        self.history = self.model.fit(
            X, y,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            validation_split=LSTM_VAL_SPLIT,
            callbacks=[cb],
            verbose=0,
        )
        return self

    def predict_insample(self, spread: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(spread.reshape(-1, 1))
        X, _ = self._make_sequences(scaled)
        preds_scaled = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(preds_scaled).flatten()

    def forecast(self, spread: np.ndarray, steps: int = FORECAST_STEPS) -> np.ndarray:
        """Итерационный прогноз на steps шагов вперёд."""
        scaled = self.scaler.transform(spread.reshape(-1, 1))
        window = scaled[-self.look_back:].reshape(1, self.look_back, 1)
        preds_scaled = []
        for _ in range(steps):
            pred = self.model.predict(window, verbose=0)[0, 0]
            preds_scaled.append(pred)
            window = np.append(window[:, 1:, :], [[[pred]]], axis=1)
        return self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()


# ──────────────────────────────────────────────
#  Multivariate (спред + VIX)
# ──────────────────────────────────────────────

class MultivariateLSTM:
    """
    Вход: [spread, vix] нормированные.
    Прогноз только spread.
    При итерационном прогнозе VIX = последнее известное значение (naive).
    """

    def __init__(self, look_back: int = LOOK_BACK):
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        self.model: tf.keras.Model | None = None
        self.history = None

    def _make_sequences(self, scaled: np.ndarray):
        """scaled shape: (T, 2) — [spread, vix]"""
        X, y = [], []
        for i in range(len(scaled) - self.look_back):
            X.append(scaled[i: i + self.look_back, :])   # (look_back, 2)
            y.append(scaled[i + self.look_back, 0])       # target = spread
        return np.array(X), np.array(y)

    def fit(self, spread: np.ndarray, vix: np.ndarray) -> "MultivariateLSTM":
        _set_seed()
        data = np.column_stack([spread, vix])
        scaled = self.scaler.fit_transform(data)
        X, y = self._make_sequences(scaled)
        self.model = _build_model(n_features=2)
        cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=LSTM_PATIENCE,
            restore_best_weights=True, verbose=0,
        )
        self.history = self.model.fit(
            X, y,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            validation_split=LSTM_VAL_SPLIT,
            callbacks=[cb],
            verbose=0,
        )
        return self

    def predict_insample(self, spread: np.ndarray, vix: np.ndarray) -> np.ndarray:
        data = np.column_stack([spread, vix])
        scaled = self.scaler.transform(data)
        X, _ = self._make_sequences(scaled)
        preds_scaled = self.model.predict(X, verbose=0)
        # Обратное преобразование только для спреда
        dummy = np.zeros((len(preds_scaled), 2))
        dummy[:, 0] = preds_scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]

    def forecast(self, spread: np.ndarray, vix: np.ndarray,
                 steps: int = FORECAST_STEPS,
                 vix_future: np.ndarray | None = None) -> np.ndarray:
        """
        Итерационный прогноз на steps шагов.
        vix_future: если None, используется последнее известное значение VIX.
        """
        if vix_future is None:
            vix_future = np.full(steps, vix[-1])

        data = np.column_stack([spread, vix])
        scaled = self.scaler.transform(data)
        window = scaled[-self.look_back:].copy()   # (look_back, 2)

        preds_scaled = []
        for i in range(steps):
            x_in = window.reshape(1, self.look_back, 2)
            pred_s = self.model.predict(x_in, verbose=0)[0, 0]
            preds_scaled.append(pred_s)

            # Следующий шаг: добавляем прогнозный спред + прогнозный VIX
            vix_norm = self.scaler.transform(
                np.array([[0.0, vix_future[i]]])
            )[0, 1]
            new_row = np.array([pred_s, vix_norm])
            window = np.vstack([window[1:], new_row])

        dummy = np.zeros((len(preds_scaled), 2))
        dummy[:, 0] = preds_scaled
        return self.scaler.inverse_transform(dummy)[:, 0]


if __name__ == "__main__":
    np.random.seed(42)
    n = 400
    spread = np.cumsum(np.random.normal(0, 0.005, n))
    vix = 20 + np.random.normal(0, 3, n)

    print("=== UnivariateLSTM ===")
    uni = UnivariateLSTM()
    uni.fit(spread)
    fc = uni.forecast(spread, steps=5)
    print("Forecast:", fc)

    print("=== MultivariateLSTM ===")
    multi = MultivariateLSTM()
    multi.fit(spread, vix)
    fc_m = multi.forecast(spread, vix, steps=5)
    print("Forecast:", fc_m)
