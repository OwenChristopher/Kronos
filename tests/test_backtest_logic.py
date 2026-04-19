from pathlib import Path

import numpy as np
import pandas as pd

import backtest
from config import Config


FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]


class FakePredictor:
    def predict_paths(
        self,
        df,
        x_timestamp,
        y_timestamp,
        pred_len,
        T,
        top_k,
        top_p,
        sample_count,
        verbose,
    ):
        previous_close = float(df["close"].iloc[-1])
        close_path = np.linspace(previous_close * 1.01, previous_close * 1.05, pred_len)
        frame = pd.DataFrame(
            {
                "open": close_path,
                "high": close_path,
                "low": close_path,
                "close": close_path,
                "volume": np.zeros(pred_len),
                "amount": np.zeros(pred_len),
            },
            index=y_timestamp,
        )
        return [frame[FEATURE_NAMES].copy() for _ in range(sample_count)]


def _build_kline_df() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    closes = [100.0, 100.0, 110.0, 121.0, 133.1]
    rows = []
    for timestamp, close in zip(timestamps, closes):
        rows.append(
            {
                "timestamps": timestamp,
                "close_time": timestamp + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
                "amount": close,
            }
        )
    return pd.DataFrame(rows)


def test_run_backtest_applies_signal_to_predicted_bar(monkeypatch, tmp_path):
    kline_df = _build_kline_df()

    monkeypatch.setattr(backtest, "load_predictor", lambda *_args, **_kwargs: FakePredictor())
    monkeypatch.setattr(
        backtest,
        "update_kline_cache",
        lambda *_args, **_kwargs: (kline_df.copy(), tmp_path / "dummy.csv"),
    )
    monkeypatch.setattr(Config, "cache_root", tmp_path)
    monkeypatch.setattr(
        Config,
        "backtest",
        {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "lookback": 2,
            "pred_len": 2,
            "evaluation_days": 2,
            "use_monte_carlo_average": False,
            "monte_carlo_paths": 1,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.95,
            "use_volume": True,
            "apply_fees": False,
            "entry_fee_rate": 0.001,
            "exit_fee_rate": 0.001,
            "entry_threshold_multiplier": 1.0,
            "exit_threshold_multiplier": 0.5,
        },
    )

    _summary, results_df, _artifact = backtest.run_backtest()

    first_bar = results_df.iloc[0]
    assert first_bar["previous_position"] == 0.0
    assert first_bar["position"] == 1.0
    assert bool(first_bar["entry_event"]) is True
    assert np.isclose(first_bar["actual_return_step1"], 0.10)
    assert np.isclose(first_bar["actual_return_horizon"], 0.21)
    assert np.isclose(first_bar["gross_strategy_return"], 0.10)


def test_compute_backtest_metrics_skips_buy_hold_fees_when_disabled():
    results_df = pd.DataFrame(
        {
            "predicted_return_step1": [0.02, -0.01],
            "actual_return_step1": [0.10, -0.05],
            "predicted_return_horizon": [0.15, -0.04],
            "actual_return_horizon": [0.12, -0.03],
            "horizon_end_up_probability": [0.70, 0.20],
            "gross_strategy_return": [0.10, -0.05],
            "net_strategy_return": [0.10, -0.05],
            "position": [1.0, 1.0],
            "entry_event": [True, False],
            "exit_event": [False, False],
            "fee_paid": [0.0, 0.0],
        }
    )

    metrics = backtest.compute_backtest_metrics(
        results_df,
        interval="1d",
        apply_fees=False,
        entry_fee=0.01,
        exit_fee=0.01,
    )

    assert np.isclose(
        metrics["cumulative_buy_hold_return_net"],
        metrics["cumulative_buy_hold_return_gross"],
    )
    assert np.isclose(metrics["next_bar_direction_accuracy"], 1.0)
    assert np.isclose(metrics["direction_accuracy"], 1.0)
    assert np.isclose(metrics["horizon_end_direction_accuracy"], 1.0)
    assert np.isclose(metrics["horizon_end_mean_direction_accuracy"], 1.0)
