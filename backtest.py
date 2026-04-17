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
        "symbol", "interval", "lookback", "evaluation_days",
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

def compute_backtest_metrics(results_df: pd.DataFrame, interval: str) -> dict:
    predicted_returns = results_df["predicted_return"].to_numpy(dtype=np.float64)
    actual_returns = results_df["actual_return"].to_numpy(dtype=np.float64)
    gross_strategy_returns = results_df["gross_strategy_return"].to_numpy(dtype=np.float64)
    net_strategy_returns = results_df["net_strategy_return"].to_numpy(dtype=np.float64)

    mae = float(np.mean(np.abs(predicted_returns - actual_returns)))
    direction_accuracy = float(np.mean(np.sign(predicted_returns) == np.sign(actual_returns)))

    correlation = None
    if len(results_df) > 1 and np.std(predicted_returns) > 0 and np.std(actual_returns) > 0:
        correlation = float(np.corrcoef(predicted_returns, actual_returns)[0, 1])

    cumulative_strategy_return_gross = float(np.prod(1.0 + gross_strategy_returns) - 1.0)
    cumulative_strategy_return_net = float(np.prod(1.0 + net_strategy_returns) - 1.0)
    cumulative_buy_hold_return_gross = float(np.prod(1.0 + actual_returns) - 1.0)

    entry_fee = float(Config.backtest["entry_fee_rate"])
    exit_fee = float(Config.backtest["exit_fee_rate"])
    buy_hold_fee_multiplier = (1.0 - entry_fee) * (1.0 - exit_fee)
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
        "direction_accuracy": direction_accuracy,
        "return_mae": mae,
        "return_correlation": correlation,
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
        min_rows=backtest_config["lookback"] + evaluation_bars,
    )
    minimum_rows = backtest_config["lookback"] + evaluation_bars
    if len(kline_df) < minimum_rows:
        raise RuntimeError(
            f"Need at least {minimum_rows} closed {interval} candles, got {len(kline_df)}."
        )

    print(
        f"Loaded {len(kline_df)} {interval} bars | "
        f"entry threshold: {entry_threshold*100:.3f}% | exit threshold: {exit_threshold*100:.3f}% | "
        f"round-trip fee: {rt_fee*100:.3f}%"
    )

    evaluation_start = len(kline_df) - evaluation_bars
    rows = []

    position = 0.0

    for target_idx in tqdm(range(evaluation_start, len(kline_df)), desc="Backtesting", unit="bar"):
        context_start = target_idx - backtest_config["lookback"]
        context_df = kline_df.iloc[context_start:target_idx].copy().reset_index(drop=True)
        actual_row = kline_df.iloc[target_idx]
        previous_close = float(context_df["close"].iloc[-1])
        actual_close = float(actual_row["close"])
        actual_return = float((actual_close / previous_close) - 1.0)

        x_timestamp = context_df["timestamps"].reset_index(drop=True)
        last_ts = context_df["timestamps"].iloc[-1]
        y_timestamp = compute_future_timestamps(last_ts, interval, 1)

        model_df = prepare_model_frame(context_df, backtest_config["use_volume"])

        with torch.no_grad():
            path_frames = predictor.predict_paths(
                df=model_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=1,
                T=backtest_config["temperature"],
                top_k=backtest_config["top_k"],
                top_p=backtest_config["top_p"],
                sample_count=effective_sample_count,
                verbose=False,
            )

        path_closes = np.array(
            [float(frame[FEATURE_NAMES].to_numpy(dtype=np.float64)[0, CLOSE_INDEX])
             for frame in path_frames]
        )
        predicted_close_mean = float(np.mean(path_closes))
        predicted_return = float((predicted_close_mean / previous_close) - 1.0)
        next_bar_up_probability = float(np.mean(path_closes > previous_close))

        prev_position = position
        new_position, entry_event, exit_event = decide_next_position(
            position, predicted_return, entry_threshold, exit_threshold,
        )
        position = new_position

        fee_paid = 0.0
        if apply_fees:
            if exit_event:
                fee_paid += exit_fee_rate
            if entry_event:
                fee_paid += entry_fee_rate

        gross_strategy_return = float(prev_position * actual_return)
        net_strategy_return = float((1.0 + gross_strategy_return) * (1.0 - fee_paid) - 1.0)

        rows.append({
            "target_timestamp": isoformat_timestamp(actual_row["timestamps"]),
            "previous_close": previous_close,
            "predicted_close": predicted_close_mean,
            "actual_close": actual_close,
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "next_bar_up_probability": next_bar_up_probability,
            "position": prev_position,
            "new_position": position,
            "entry_event": entry_event,
            "exit_event": exit_event,
            "fee_paid": fee_paid,
            "gross_strategy_return": gross_strategy_return,
            "net_strategy_return": net_strategy_return,
            "forecast_mode": forecast_mode,
            "effective_sample_count": effective_sample_count,
            "close_abs_error": float(abs(predicted_close_mean - actual_close)),
        })

    results_df = pd.DataFrame(rows)
    metrics = compute_backtest_metrics(results_df, interval)

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
        f"mode: {p['forecast_mode']}"
    )
    print(
        f"entry: {p['entry_threshold']*100:.3f}% | exit: {p['exit_threshold']*100:.3f}% | "
        f"round-trip fee: {p['round_trip_fee_rate']*100:.3f}%"
    )
    print(f"{'─'*80}")
    print(f"  Prediction quality:")
    print(
        f"    direction accuracy: {m['direction_accuracy']:.1%} | "
        f"return MAE: {m['return_mae']:.6f} | "
        f"correlation: {m['return_correlation'] if m['return_correlation'] is not None else 'n/a'}"
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
