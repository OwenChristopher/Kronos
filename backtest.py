import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from inference import (
    compute_future_timestamps,
    interval_to_timedelta,
    isoformat_timestamp,
    load_predictor,
    prepare_model_frame,
    update_kline_cache,
)


FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]
CLOSE_INDEX = FEATURE_NAMES.index("close")


def get_backtest_config() -> dict:
    config = dict(Config.backtest)
    required_keys = {
        "symbol", "interval", "lookback", "pred_len", "evaluation_days",
        "use_monte_carlo_average", "monte_carlo_paths",
        "temperature", "top_k", "top_p", "use_volume",
        "apply_fees", "entry_fee_rate", "exit_fee_rate",
        "entry_threshold_multiplier", "exit_threshold_multiplier",
    }
    missing = sorted(required_keys.difference(config))
    if missing:
        raise KeyError(f"Missing backtest config keys in config.py: {missing}")
    if config["monte_carlo_paths"] < 1:
        raise ValueError("monte_carlo_paths must be at least 1.")
    if int(config["pred_len"]) < 1:
        raise ValueError("pred_len must be at least 1.")
    if float(config["evaluation_days"]) <= 0:
        raise ValueError("evaluation_days must be greater than 0.")
    return config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_evaluation_bars(interval: str, evaluation_days: int | float) -> int:
    interval_delta = interval_to_timedelta(interval)
    evaluation_span = pd.Timedelta(days=float(evaluation_days))
    return max(1, math.ceil(evaluation_span / interval_delta))


def round_trip_fee(entry_fee: float, exit_fee: float) -> float:
    return 1.0 - (1.0 - entry_fee) * (1.0 - exit_fee)


# ---------------------------------------------------------------------------
# Position state machine
# ---------------------------------------------------------------------------
#
# Fee-aware threshold with hysteresis:
#   - Enter only when |predicted_return| exceeds entry_threshold
#   - Stay in position while opposite signal is weaker than exit_threshold
#   - Flip in a single step when opposite signal clears entry_threshold
#
# entry_threshold > exit_threshold. Gap creates the no-trade band that kills
# the flip-flop fee drag.

def decide_next_position(
    current_position: float,
    predicted_return: float,
    entry_threshold: float,
    exit_threshold: float,
) -> tuple[float, bool, bool]:
    """Return (new_position, entry_event, exit_event)."""
    if current_position == 0.0:
        if predicted_return > entry_threshold:
            return 1.0, True, False
        if predicted_return < -entry_threshold:
            return -1.0, True, False
        return 0.0, False, False

    if current_position > 0.0:
        if predicted_return < -entry_threshold:
            return -1.0, True, True  # flip
        if predicted_return < -exit_threshold:
            return 0.0, False, True  # exit to flat
        return current_position, False, False

    # current_position < 0
    if predicted_return > entry_threshold:
        return 1.0, True, True  # flip
    if predicted_return > exit_threshold:
        return 0.0, False, True  # exit to flat
    return current_position, False, False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_backtest_metrics(
    results_df: pd.DataFrame,
    interval: str,
    apply_fees: bool,
    entry_fee: float,
    exit_fee: float,
) -> dict:
    predicted_returns_step1 = results_df["predicted_return_step1"].to_numpy(dtype=np.float64)
    actual_returns_step1 = results_df["actual_return_step1"].to_numpy(dtype=np.float64)
    predicted_returns_horizon = results_df["predicted_return_horizon"].to_numpy(dtype=np.float64)
    actual_returns_horizon = results_df["actual_return_horizon"].to_numpy(dtype=np.float64)
    gross_strategy_returns = results_df["gross_strategy_return"].to_numpy(dtype=np.float64)
    net_strategy_returns = results_df["net_strategy_return"].to_numpy(dtype=np.float64)

    next_bar_return_mae = float(np.mean(np.abs(predicted_returns_step1 - actual_returns_step1)))
    horizon_return_mae = float(np.mean(np.abs(predicted_returns_horizon - actual_returns_horizon)))
    next_bar_direction_accuracy = float(
        np.mean(np.sign(predicted_returns_step1) == np.sign(actual_returns_step1))
    )
    horizon_end_direction_accuracy = float(
        np.mean((results_df["horizon_end_up_probability"].to_numpy(dtype=np.float64) >= 0.5) ==
                (actual_returns_horizon >= 0.0))
    )
    horizon_end_mean_direction_accuracy = float(
        np.mean(np.sign(predicted_returns_horizon) == np.sign(actual_returns_horizon))
    )

    next_bar_return_correlation = None
    if (
        len(results_df) > 1
        and np.std(predicted_returns_step1) > 0
        and np.std(actual_returns_step1) > 0
    ):
        next_bar_return_correlation = float(
            np.corrcoef(predicted_returns_step1, actual_returns_step1)[0, 1]
        )

    horizon_return_correlation = None
    if (
        len(results_df) > 1
        and np.std(predicted_returns_horizon) > 0
        and np.std(actual_returns_horizon) > 0
    ):
        horizon_return_correlation = float(
            np.corrcoef(predicted_returns_horizon, actual_returns_horizon)[0, 1]
        )

    cumulative_strategy_return_gross = float(np.prod(1.0 + gross_strategy_returns) - 1.0)
    cumulative_strategy_return_net = float(np.prod(1.0 + net_strategy_returns) - 1.0)
    cumulative_buy_hold_return_gross = float(np.prod(1.0 + actual_returns_step1) - 1.0)

    buy_hold_fee_multiplier = (1.0 - entry_fee) * (1.0 - exit_fee) if apply_fees else 1.0
    cumulative_buy_hold_return_net = float(
        (1.0 + cumulative_buy_hold_return_gross) * buy_hold_fee_multiplier - 1.0
    )

    if np.std(net_strategy_returns) > 0:
        interval_hours = interval_to_timedelta(interval).total_seconds() / 3600
        bars_per_year = 365.25 * 24 / interval_hours
        sharpe = float(
            np.mean(net_strategy_returns) / np.std(net_strategy_returns) * np.sqrt(bars_per_year)
        )
    else:
        sharpe = 0.0

    cumulative = np.cumprod(1.0 + net_strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    entries = int(results_df["entry_event"].sum())
    exits = int(results_df["exit_event"].sum())
    bars_in_position = int((results_df["position"] != 0).sum())

    return {
        "num_periods": int(len(results_df)),
        "direction_accuracy": horizon_end_direction_accuracy,
        "next_bar_direction_accuracy": next_bar_direction_accuracy,
        "horizon_end_direction_accuracy": horizon_end_direction_accuracy,
        "horizon_end_mean_direction_accuracy": horizon_end_mean_direction_accuracy,
        "next_bar_return_mae": next_bar_return_mae,
        "horizon_return_mae": horizon_return_mae,
        "return_mae": horizon_return_mae,
        "next_bar_return_correlation": next_bar_return_correlation,
        "horizon_return_correlation": horizon_return_correlation,
        "return_correlation": horizon_return_correlation,
        "cumulative_strategy_return_gross": cumulative_strategy_return_gross,
        "cumulative_strategy_return_net": cumulative_strategy_return_net,
        "cumulative_buy_hold_return_gross": cumulative_buy_hold_return_gross,
        "cumulative_buy_hold_return_net": cumulative_buy_hold_return_net,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "entry_count": entries,
        "exit_count": exits,
        "round_trips": min(entries, exits),
        "bars_in_position": bars_in_position,
        "bars_flat": int(len(results_df)) - bars_in_position,
        "total_fees_paid": float(results_df["fee_paid"].sum()),
    }


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest() -> tuple[dict, pd.DataFrame, Path]:
    backtest_config = get_backtest_config()
    model_config = dict(Config.backtest_model)
    predictor = load_predictor(Config, model_config)
    symbol = backtest_config["symbol"]
    interval = backtest_config["interval"]
    effective_sample_count = (
        backtest_config["monte_carlo_paths"]
        if backtest_config["use_monte_carlo_average"]
        else 1
    )
    evaluation_bars = compute_evaluation_bars(interval, backtest_config["evaluation_days"])
    pred_len = int(backtest_config["pred_len"])
    forecast_mode = (
        f"monte_carlo_average_{backtest_config['monte_carlo_paths']}"
        if backtest_config["use_monte_carlo_average"]
        else "single_path"
    )

    entry_fee_rate = float(backtest_config["entry_fee_rate"])
    exit_fee_rate = float(backtest_config["exit_fee_rate"])
    apply_fees = backtest_config["apply_fees"]
    rt_fee = round_trip_fee(entry_fee_rate, exit_fee_rate)
    entry_threshold = float(backtest_config["entry_threshold_multiplier"]) * rt_fee
    exit_threshold = float(backtest_config["exit_threshold_multiplier"]) * rt_fee
    if exit_threshold >= entry_threshold:
        raise ValueError(
            "exit_threshold_multiplier must be < entry_threshold_multiplier for hysteresis."
        )

    kline_df, cache_path = update_kline_cache(
        Config.cache_root, symbol, interval, Config.binance_limit,
        min_rows=backtest_config["lookback"] + evaluation_bars + pred_len - 1,
    )
    minimum_rows = backtest_config["lookback"] + evaluation_bars + pred_len - 1
    if len(kline_df) < minimum_rows:
        raise RuntimeError(
            f"Need at least {minimum_rows} closed {interval} candles, got {len(kline_df)}."
        )

    print(
        f"Loaded {len(kline_df)} {interval} bars | "
        f"forecast horizon: {pred_len} bars | "
        f"entry threshold: {entry_threshold*100:.3f}% | exit threshold: {exit_threshold*100:.3f}% | "
        f"round-trip fee: {rt_fee*100:.3f}%"
    )

    evaluation_stop = len(kline_df) - pred_len + 1
    evaluation_start = evaluation_stop - evaluation_bars
    rows = []

    position = 0.0

    for target_idx in tqdm(range(evaluation_start, evaluation_stop), desc="Backtesting", unit="bar"):
        context_start = target_idx - backtest_config["lookback"]
        context_df = kline_df.iloc[context_start:target_idx].copy().reset_index(drop=True)
        actual_row_step1 = kline_df.iloc[target_idx]
        actual_row_horizon = kline_df.iloc[target_idx + pred_len - 1]
        previous_close = float(context_df["close"].iloc[-1])
        actual_close_step1 = float(actual_row_step1["close"])
        actual_close_horizon = float(actual_row_horizon["close"])
        actual_return_step1 = float((actual_close_step1 / previous_close) - 1.0)
        actual_return_horizon = float((actual_close_horizon / previous_close) - 1.0)

        x_timestamp = context_df["timestamps"].reset_index(drop=True)
        last_ts = context_df["timestamps"].iloc[-1]
        y_timestamp = compute_future_timestamps(last_ts, interval, pred_len)

        model_df = prepare_model_frame(context_df, backtest_config["use_volume"])

        with torch.no_grad():
            path_frames = predictor.predict_paths(
                df=model_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=backtest_config["temperature"],
                top_k=backtest_config["top_k"],
                top_p=backtest_config["top_p"],
                sample_count=effective_sample_count,
                verbose=False,
            )

        path_closes = np.stack(
            [frame["close"].to_numpy(dtype=np.float64) for frame in path_frames]
        )
        path_closes_step1 = path_closes[:, 0]
        path_closes_horizon = path_closes[:, -1]
        predicted_close_step1 = float(np.mean(path_closes_step1))
        predicted_close_horizon = float(np.mean(path_closes_horizon))
        predicted_return_step1 = float((predicted_close_step1 / previous_close) - 1.0)
        predicted_return_horizon = float((predicted_close_horizon / previous_close) - 1.0)
        next_bar_up_probability = float(np.mean(path_closes_step1 > previous_close))
        horizon_end_up_probability = float(np.mean(path_closes_horizon > previous_close))

        prev_position = position
        new_position, entry_event, exit_event = decide_next_position(
            position, predicted_return_horizon, entry_threshold, exit_threshold,
        )
        position = new_position

        fee_paid = 0.0
        if apply_fees:
            if exit_event:
                fee_paid += exit_fee_rate
            if entry_event:
                fee_paid += entry_fee_rate

        # The forecast is formed before the target bar opens, so the chosen
        # position is the exposure for that bar's realized return.
        gross_strategy_return = float(position * actual_return_step1)
        net_strategy_return = float((1.0 + gross_strategy_return) * (1.0 - fee_paid) - 1.0)

        rows.append({
            "target_timestamp": isoformat_timestamp(actual_row_step1["timestamps"]),
            "horizon_end_timestamp": isoformat_timestamp(actual_row_horizon["timestamps"]),
            "previous_close": previous_close,
            "predicted_close_step1": predicted_close_step1,
            "predicted_close_horizon": predicted_close_horizon,
            "actual_close_step1": actual_close_step1,
            "actual_close_horizon": actual_close_horizon,
            "predicted_return_step1": predicted_return_step1,
            "predicted_return_horizon": predicted_return_horizon,
            "actual_return_step1": actual_return_step1,
            "actual_return_horizon": actual_return_horizon,
            "signal_return": predicted_return_horizon,
            "next_bar_up_probability": next_bar_up_probability,
            "horizon_end_up_probability": horizon_end_up_probability,
            "previous_position": prev_position,
            "position": position,
            "new_position": position,
            "entry_event": entry_event,
            "exit_event": exit_event,
            "fee_paid": fee_paid,
            "gross_strategy_return": gross_strategy_return,
            "net_strategy_return": net_strategy_return,
            "forecast_mode": forecast_mode,
            "effective_sample_count": effective_sample_count,
            "close_abs_error_step1": float(abs(predicted_close_step1 - actual_close_step1)),
            "close_abs_error_horizon": float(abs(predicted_close_horizon - actual_close_horizon)),
        })

    results_df = pd.DataFrame(rows)
    metrics = compute_backtest_metrics(
        results_df,
        interval,
        apply_fees=apply_fees,
        entry_fee=entry_fee_rate,
        exit_fee=exit_fee_rate,
    )

    generated_at = datetime.now(UTC)
    run_id = f"{symbol.lower()}_{interval}_backtest_{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    artifact_dir = ensure_dir(Config.cache_root / "backtests" / symbol.upper() / interval)
    csv_path = artifact_dir / f"{run_id}.csv"
    json_path = artifact_dir / f"{run_id}.json"

    results_df.to_csv(csv_path, index=False)
    summary = {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "symbol": symbol,
        "interval": interval,
        "model": model_config,
        "kline_cache_path": str(cache_path),
        "params": {
            **backtest_config,
            "evaluation_bars": evaluation_bars,
            "effective_sample_count": effective_sample_count,
            "forecast_mode": forecast_mode,
            "round_trip_fee_rate": rt_fee,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
        },
        "metrics": metrics,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary, results_df, json_path


def print_summary(summary: dict) -> None:
    m = summary["metrics"]
    p = summary["params"]
    print(f"\nBacktest saved: {summary['artifacts']['json']}")
    print(f"{'='*80}")
    print(
        f"{summary['symbol']} {summary['interval']} | "
        f"days: {p['evaluation_days']} | bars: {p['evaluation_bars']} | "
        f"horizon: {p['pred_len']} bars | mode: {p['forecast_mode']}"
    )
    print(
        f"entry: {p['entry_threshold']*100:.3f}% | exit: {p['exit_threshold']*100:.3f}% | "
        f"round-trip fee: {p['round_trip_fee_rate']*100:.3f}%"
    )
    print(f"{'─'*80}")
    print(f"  Prediction quality:")
    print(
        f"    horizon-end direction accuracy: {m['horizon_end_direction_accuracy']:.1%} | "
        f"mean-sign accuracy: {m['horizon_end_mean_direction_accuracy']:.1%} | "
        f"next-bar direction accuracy: {m['next_bar_direction_accuracy']:.1%}"
    )
    print(
        f"    horizon return MAE: {m['horizon_return_mae']:.6f} | "
        f"next-bar return MAE: {m['next_bar_return_mae']:.6f}"
    )
    print(
        f"    horizon correlation: "
        f"{m['horizon_return_correlation'] if m['horizon_return_correlation'] is not None else 'n/a'} | "
        f"next-bar correlation: "
        f"{m['next_bar_return_correlation'] if m['next_bar_return_correlation'] is not None else 'n/a'}"
    )
    print(f"{'─'*80}")
    print(f"  Strategy performance:")
    print(f"    gross: {m['cumulative_strategy_return_gross']:+.2%} | net: {m['cumulative_strategy_return_net']:+.2%}")
    print(
        f"    buy & hold gross: {m['cumulative_buy_hold_return_gross']:+.2%} | "
        f"net: {m['cumulative_buy_hold_return_net']:+.2%}"
    )
    print(f"    annualized sharpe: {m['annualized_sharpe']:.2f} | max drawdown: {m['max_drawdown']:.2%}")
    print(f"{'─'*80}")
    print(f"  Trading activity:")
    print(
        f"    entries: {m['entry_count']} | exits: {m['exit_count']} | "
        f"round trips: {m['round_trips']}"
    )
    print(
        f"    bars in position: {m['bars_in_position']}/{m['num_periods']} | "
        f"bars flat: {m['bars_flat']}/{m['num_periods']}"
    )
    print(f"    total fees: {m['total_fees_paid']:.6f}")
    print(f"{'='*80}\n")


def main() -> int:
    try:
        summary, _, _ = run_backtest()
    except Exception as exc:
        print(f"Backtest failed: {exc}")
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
