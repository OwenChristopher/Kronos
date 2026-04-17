import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

from config import Config
from inference import (
    aggregate_paths,
    compute_future_timestamps,
    interval_to_timedelta,
    isoformat_timestamp,
    load_predictor,
    prepare_model_frame,
    update_kline_cache,
)


FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]


def get_backtest_config() -> dict:
    config = dict(Config.backtest)
    required_keys = {
        "symbol", "interval", "lookback", "evaluation_days", "pred_len",
        "use_monte_carlo_average", "monte_carlo_paths", "temperature",
        "top_k", "top_p", "use_volume", "apply_fees",
        "entry_fee_rate", "exit_fee_rate",
        "confidence_threshold", "min_hold_bars",
        "signal_scale", "min_predicted_sharpe", "skew_weight",
        "ensemble_interval", "ensemble_lookback", "ensemble_pred_len", "ensemble_weight",
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


# ---------------------------------------------------------------------------
# Signal computation from MC paths
# ---------------------------------------------------------------------------

def compute_signal(
    path_arrays: np.ndarray,
    previous_close: float,
    pred_len: int,
    signal_scale: float,
    min_predicted_sharpe: float,
    skew_weight: float,
    confidence_threshold: float,
) -> dict:
    """
    Compute a composite trading signal from Monte Carlo path predictions.

    path_arrays: shape (num_paths, pred_len, 6) — OHLCVA per path per step
    Returns dict with signal components and final composite signal in [-1, 1].
    """
    num_paths = path_arrays.shape[0]

    # --- (E) Multi-step trajectory signal ---
    # Average predicted close across the full horizon, per path
    close_idx = FEATURE_NAMES.index("close")
    high_idx = FEATURE_NAMES.index("high")
    low_idx = FEATURE_NAMES.index("low")

    close_paths = path_arrays[:, :, close_idx]            # (num_paths, pred_len)
    high_paths = path_arrays[:, :, high_idx]
    low_paths = path_arrays[:, :, low_idx]

    # Eq. 13 from paper: average predicted close over horizon
    trajectory_mean_per_path = np.mean(close_paths, axis=1)   # (num_paths,)
    trajectory_return_per_path = (trajectory_mean_per_path / previous_close) - 1.0
    trajectory_return = float(np.mean(trajectory_return_per_path))

    # Next-bar close paths (for confidence)
    next_close_paths = close_paths[:, 0]
    up_probability = float(np.mean(next_close_paths > previous_close))
    directional_confidence = max(up_probability, 1.0 - up_probability)

    # --- (F) Volatility-adjusted signal (predicted Sharpe) ---
    # Use mean predicted high-low range as volatility proxy
    range_per_step = high_paths - low_paths                   # (num_paths, pred_len)
    mean_range = float(np.mean(range_per_step))
    predicted_vol = mean_range / previous_close
    predicted_vol = max(predicted_vol, 1e-8)
    predicted_sharpe = abs(trajectory_return) / predicted_vol

    # --- (G) MC distribution shape ---
    path_returns = trajectory_return_per_path
    skewness = float(sp_stats.skew(path_returns)) if num_paths > 2 else 0.0
    kurtosis = float(sp_stats.kurtosis(path_returns, fisher=True)) if num_paths > 2 else 0.0
    return_std = float(np.std(path_returns))

    # Tail risk: expected shortfall below p10 and above p90
    p10 = np.percentile(path_returns, 10)
    p90 = np.percentile(path_returns, 90)
    downside_tail = float(np.mean(path_returns[path_returns <= p10])) if np.any(path_returns <= p10) else 0.0
    upside_tail = float(np.mean(path_returns[path_returns >= p90])) if np.any(path_returns >= p90) else 0.0

    # --- (K) Continuous position sizing ---
    # Magnitude signal: scale predicted return
    magnitude_signal = np.clip(trajectory_return / signal_scale, -1.0, 1.0)

    # Probability signal: how one-sided is the MC distribution
    probability_signal = (up_probability - 0.5) * 2.0  # [-1, 1]

    # Skew bonus: if distribution is skewed in the direction of the trade, boost
    direction = np.sign(trajectory_return)
    skew_alignment = direction * skewness
    skew_bonus = np.clip(skew_alignment * skew_weight, -0.3, 0.3)

    # Volatility gate: penalize when predicted Sharpe is too low
    vol_gate = 1.0 if predicted_sharpe >= min_predicted_sharpe else predicted_sharpe / min_predicted_sharpe

    # Composite signal
    raw_signal = magnitude_signal * abs(probability_signal) * vol_gate + skew_bonus
    composite_signal = float(np.clip(raw_signal, -1.0, 1.0))

    return {
        "trajectory_return": trajectory_return,
        "up_probability": up_probability,
        "directional_confidence": directional_confidence,
        "predicted_vol": predicted_vol,
        "predicted_sharpe": predicted_sharpe,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "return_std": return_std,
        "downside_tail": downside_tail,
        "upside_tail": upside_tail,
        "magnitude_signal": float(magnitude_signal),
        "probability_signal": float(probability_signal),
        "skew_bonus": float(skew_bonus),
        "vol_gate": float(vol_gate),
        "composite_signal": composite_signal,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_backtest_metrics(results_df: pd.DataFrame) -> dict:
    actual_returns = results_df["actual_return"].to_numpy(dtype=np.float64)
    gross_strategy_returns = results_df["gross_strategy_return"].to_numpy(dtype=np.float64)
    net_strategy_returns = results_df["net_strategy_return"].to_numpy(dtype=np.float64)

    trajectory_returns = results_df["trajectory_return"].to_numpy(dtype=np.float64)
    direction_accuracy = float(
        np.mean(np.sign(trajectory_returns) == np.sign(actual_returns))
    )

    correlation = None
    if len(results_df) > 1 and np.std(trajectory_returns) > 0 and np.std(actual_returns) > 0:
        correlation = float(np.corrcoef(trajectory_returns, actual_returns)[0, 1])

    cumulative_strategy_return_gross = float(np.prod(1.0 + gross_strategy_returns) - 1.0)
    cumulative_strategy_return_net = float(np.prod(1.0 + net_strategy_returns) - 1.0)
    cumulative_buy_hold_return_gross = float(np.prod(1.0 + actual_returns) - 1.0)

    buy_hold_fee_multiplier = 1.0
    if len(results_df) > 0:
        entry_fee = float(Config.backtest["entry_fee_rate"])
        exit_fee = float(Config.backtest["exit_fee_rate"])
        buy_hold_fee_multiplier = (1.0 - entry_fee) * (1.0 - exit_fee)
    cumulative_buy_hold_return_net = float(
        (1.0 + cumulative_buy_hold_return_gross) * buy_hold_fee_multiplier - 1.0
    )

    # Sharpe ratio (annualized)
    if np.std(net_strategy_returns) > 0:
        interval_hours = interval_to_timedelta(Config.backtest["interval"]).total_seconds() / 3600
        bars_per_year = 365.25 * 24 / interval_hours
        sharpe = float(
            np.mean(net_strategy_returns) / np.std(net_strategy_returns) * np.sqrt(bars_per_year)
        )
    else:
        sharpe = 0.0

    # Max drawdown
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
        "average_position_size": float(results_df["position"].abs().mean()),
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
    pred_len = int(backtest_config["pred_len"])
    effective_sample_count = (
        backtest_config["monte_carlo_paths"]
        if backtest_config["use_monte_carlo_average"]
        else 1
    )
    evaluation_bars = compute_evaluation_bars(
        interval, backtest_config["evaluation_days"],
    )
    forecast_mode = (
        f"monte_carlo_average_{backtest_config['monte_carlo_paths']}"
        if backtest_config["use_monte_carlo_average"]
        else "single_path"
    )

    confidence_threshold = float(backtest_config["confidence_threshold"])
    min_hold_bars = int(backtest_config["min_hold_bars"])
    signal_scale = float(backtest_config["signal_scale"])
    min_predicted_sharpe = float(backtest_config["min_predicted_sharpe"])
    skew_weight = float(backtest_config["skew_weight"])
    entry_fee_rate = float(backtest_config["entry_fee_rate"])
    exit_fee_rate = float(backtest_config["exit_fee_rate"])
    apply_fees = backtest_config["apply_fees"]

    # --- Primary timeframe data ---
    kline_df, cache_path = update_kline_cache(
        Config.cache_root, symbol, interval, Config.binance_limit,
        min_rows=backtest_config["lookback"] + evaluation_bars,
    )
    minimum_rows = backtest_config["lookback"] + evaluation_bars
    if len(kline_df) < minimum_rows:
        raise RuntimeError(
            f"Need at least {minimum_rows} closed {interval} candles, got {len(kline_df)}."
        )

    # --- (L) Secondary timeframe data for ensemble ---
    ensemble_interval = backtest_config["ensemble_interval"]
    ensemble_lookback = int(backtest_config["ensemble_lookback"])
    ensemble_pred_len = int(backtest_config["ensemble_pred_len"])
    ensemble_weight = float(backtest_config["ensemble_weight"])

    ensemble_kline_df, _ = update_kline_cache(
        Config.cache_root, symbol, ensemble_interval, Config.binance_limit,
        min_rows=ensemble_lookback + evaluation_bars * 4,
    )
    print(
        f"Loaded {len(kline_df)} {interval} bars + {len(ensemble_kline_df)} {ensemble_interval} bars for ensemble"
    )

    evaluation_start = len(kline_df) - evaluation_bars
    rows = []

    position = 0.0
    bars_held = 0

    for target_idx in range(evaluation_start, len(kline_df)):
        context_start = target_idx - backtest_config["lookback"]
        context_df = kline_df.iloc[context_start:target_idx].copy().reset_index(drop=True)
        actual_row = kline_df.iloc[target_idx]
        previous_close = float(context_df["close"].iloc[-1])
        actual_close = float(actual_row["close"])
        actual_return = float((actual_close / previous_close) - 1.0)

        x_timestamp = context_df["timestamps"].reset_index(drop=True)

        # Build future timestamps for multi-step prediction
        last_ts = context_df["timestamps"].iloc[-1]
        y_timestamp = compute_future_timestamps(last_ts, interval, pred_len)

        model_df = prepare_model_frame(context_df, backtest_config["use_volume"])

        # --- Primary timeframe prediction ---
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

        path_arrays = np.stack(
            [frame[FEATURE_NAMES].to_numpy(dtype=np.float64) for frame in path_frames]
        )

        sig = compute_signal(
            path_arrays, previous_close, pred_len,
            signal_scale, min_predicted_sharpe, skew_weight, confidence_threshold,
        )

        # --- (L) Ensemble: secondary timeframe prediction ---
        ensemble_signal = 0.0
        primary_bar_ts = pd.Timestamp(last_ts)
        if primary_bar_ts.tzinfo is None:
            primary_bar_ts = primary_bar_ts.tz_localize(UTC)
        else:
            primary_bar_ts = primary_bar_ts.tz_convert(UTC)

        ens_mask = ensemble_kline_df["timestamps"].apply(
            lambda t: (pd.Timestamp(t).tz_convert(UTC) if pd.Timestamp(t).tzinfo else pd.Timestamp(t).tz_localize(UTC)) <= primary_bar_ts
        )
        ens_available = ensemble_kline_df[ens_mask]

        if len(ens_available) >= ensemble_lookback:
            ens_context = ens_available.tail(ensemble_lookback).copy().reset_index(drop=True)
            ens_x_ts = ens_context["timestamps"].reset_index(drop=True)
            ens_last_ts = ens_context["timestamps"].iloc[-1]
            ens_y_ts = compute_future_timestamps(ens_last_ts, ensemble_interval, ensemble_pred_len)
            ens_model_df = prepare_model_frame(ens_context, backtest_config["use_volume"])

            with torch.no_grad():
                ens_path_frames = predictor.predict_paths(
                    df=ens_model_df,
                    x_timestamp=ens_x_ts,
                    y_timestamp=ens_y_ts,
                    pred_len=ensemble_pred_len,
                    T=backtest_config["temperature"],
                    top_k=backtest_config["top_k"],
                    top_p=backtest_config["top_p"],
                    sample_count=effective_sample_count,
                    verbose=False,
                )

            ens_arrays = np.stack(
                [f[FEATURE_NAMES].to_numpy(dtype=np.float64) for f in ens_path_frames]
            )
            ens_sig = compute_signal(
                ens_arrays, previous_close, ensemble_pred_len,
                signal_scale, min_predicted_sharpe, skew_weight, confidence_threshold,
            )
            ensemble_signal = ens_sig["composite_signal"]

        # Blend: primary * (1 - weight) + ensemble * weight
        blended_signal = sig["composite_signal"] * (1.0 - ensemble_weight) + ensemble_signal * ensemble_weight
        blended_signal = float(np.clip(blended_signal, -1.0, 1.0))

        # Confluence check: if primary and ensemble disagree on direction, dampen
        if np.sign(sig["composite_signal"]) != np.sign(ensemble_signal) and ensemble_signal != 0.0:
            blended_signal *= 0.3

        # Position management with continuous sizing
        prev_position = position
        entry_event = False
        exit_event = False
        fee_paid = 0.0

        desired_position = blended_signal
        confident = sig["directional_confidence"] >= confidence_threshold

        if position == 0.0:
            if confident and abs(desired_position) > 0.1:
                position = desired_position
                bars_held = 0
                entry_event = True
                if apply_fees:
                    fee_paid += entry_fee_rate
        else:
            bars_held += 1
            can_exit = bars_held >= min_hold_bars

            if can_exit:
                # Position flip: sign changed and new signal is confident
                sign_flipped = np.sign(desired_position) != np.sign(position) and np.sign(desired_position) != 0
                conviction_gone = not confident or abs(desired_position) < 0.05

                if sign_flipped and confident and abs(desired_position) > 0.1:
                    exit_event = True
                    entry_event = True
                    position = desired_position
                    bars_held = 0
                    if apply_fees:
                        fee_paid += exit_fee_rate + entry_fee_rate
                elif conviction_gone:
                    exit_event = True
                    position = 0.0
                    bars_held = 0
                    if apply_fees:
                        fee_paid += exit_fee_rate
                else:
                    # Update position size without paying fees (same direction)
                    position = desired_position

        gross_strategy_return = float(prev_position * actual_return)
        net_strategy_return = gross_strategy_return - fee_paid

        # First-step predicted close (for logging)
        mean_first_close = float(np.mean(path_arrays[:, 0, FEATURE_NAMES.index("close")]))

        rows.append({
            "target_timestamp": isoformat_timestamp(actual_row["timestamps"]),
            "previous_close": previous_close,
            "predicted_close_step1": mean_first_close,
            "actual_close": actual_close,
            "actual_return": actual_return,
            "trajectory_return": sig["trajectory_return"],
            "up_probability": sig["up_probability"],
            "directional_confidence": sig["directional_confidence"],
            "predicted_vol": sig["predicted_vol"],
            "predicted_sharpe": sig["predicted_sharpe"],
            "skewness": sig["skewness"],
            "kurtosis": sig["kurtosis"],
            "return_std": sig["return_std"],
            "downside_tail": sig["downside_tail"],
            "upside_tail": sig["upside_tail"],
            "magnitude_signal": sig["magnitude_signal"],
            "probability_signal": sig["probability_signal"],
            "skew_bonus": sig["skew_bonus"],
            "vol_gate": sig["vol_gate"],
            "composite_signal": sig["composite_signal"],
            "ensemble_signal": ensemble_signal,
            "blended_signal": blended_signal,
            "position": prev_position,
            "new_position": position,
            "entry_event": entry_event,
            "exit_event": exit_event,
            "bars_held": bars_held,
            "fee_paid": fee_paid,
            "entry_fee_rate": entry_fee_rate if entry_event else 0.0,
            "exit_fee_rate": exit_fee_rate if exit_event else 0.0,
            "gross_strategy_return": gross_strategy_return,
            "net_strategy_return": net_strategy_return,
            "forecast_mode": forecast_mode,
            "effective_sample_count": effective_sample_count,
            "close_abs_error": float(abs(mean_first_close - actual_close)),
        })

    results_df = pd.DataFrame(rows)
    metrics = compute_backtest_metrics(results_df)

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
        f"pred_len: {p['pred_len']} | mode: {p['forecast_mode']}"
    )
    print(f"{'─'*80}")
    print(f"  Prediction quality:")
    print(
        f"    direction accuracy: {m['direction_accuracy']:.1%} | "
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
    print(f"    avg position size: {m['average_position_size']:.3f} | total fees: {m['total_fees_paid']:.6f}")
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
