import json
import math
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Config
from model import Kronos, KronosPredictor, KronosTokenizer


FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]
CLOSE_INDEX = FEATURE_NAMES.index("close")
VOLUME_INDEX = FEATURE_NAMES.index("volume")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def isoformat_timestamp(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return ts.isoformat()


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    units = {
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
    }
    suffix = interval[-1]
    if suffix not in units:
        raise ValueError(f"Unsupported interval suffix: {interval}")
    value = int(interval[:-1])
    return pd.Timedelta(**{units[suffix]: value})


def fetch_binance_klines(
    symbol: str,
    interval: str,
    limit: int,
    end_time_ms: int | None = None,
) -> pd.DataFrame:
    params_dict = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params_dict["endTime"] = end_time_ms
    params = urllib.parse.urlencode(params_dict)
    url = f"https://api.binance.com/api/v3/klines?{params}"

    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(payload, columns=columns)
    if df.empty:
        raise RuntimeError(f"No kline data returned for {symbol} {interval}.")

    df = df[
        [
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
        ]
    ].copy()
    df.rename(
        columns={
            "open_time": "timestamps",
            "quote_asset_volume": "amount",
        },
        inplace=True,
    )
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    now_utc = pd.Timestamp(datetime.now(UTC))
    df = df[df["close_time"] < now_utc].copy()
    if df.empty:
        raise RuntimeError(f"Fetched only open candles for {symbol} {interval}.")

    df.sort_values("timestamps", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_kline_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["timestamps", "close_time"] + FEATURE_NAMES)

    df = pd.read_csv(cache_path, parse_dates=["timestamps", "close_time"])
    if "timestamps" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    for column in FEATURE_NAMES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("timestamps", inplace=True)
    df.drop_duplicates(subset=["timestamps"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def merge_kline_frames(existing_df: pd.DataFrame, fetched_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        combined = fetched_df.copy()
    elif fetched_df.empty:
        combined = existing_df.copy()
    else:
        combined = pd.concat([existing_df, fetched_df], ignore_index=True)
    combined.sort_values("timestamps", inplace=True)
    combined.drop_duplicates(subset=["timestamps"], keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def update_kline_cache(
    cache_root: Path,
    symbol: str,
    interval: str,
    limit: int,
    min_rows: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    kline_dir = ensure_dir(cache_root / "klines" / symbol.upper())
    cache_path = kline_dir / f"{interval}.csv"
    existing_df = load_kline_cache(cache_path)
    fetched_df = fetch_binance_klines(symbol, interval, limit)
    combined = merge_kline_frames(existing_df, fetched_df)

    if min_rows is not None and min_rows > 0:
        while len(combined) < min_rows:
            oldest_timestamp = pd.Timestamp(combined["timestamps"].iloc[0])
            if oldest_timestamp.tzinfo is None:
                oldest_timestamp = oldest_timestamp.tz_localize(UTC)
            else:
                oldest_timestamp = oldest_timestamp.tz_convert(UTC)
            older_end_time_ms = int(oldest_timestamp.timestamp() * 1000) - 1
            older_df = fetch_binance_klines(
                symbol,
                interval,
                limit,
                end_time_ms=older_end_time_ms,
            )
            if older_df.empty:
                break
            older_min_timestamp = pd.Timestamp(older_df["timestamps"].min())
            if older_min_timestamp.tzinfo is None:
                older_min_timestamp = older_min_timestamp.tz_localize(UTC)
            else:
                older_min_timestamp = older_min_timestamp.tz_convert(UTC)
            if older_min_timestamp >= oldest_timestamp:
                break
            combined = merge_kline_frames(combined, older_df)

    combined.to_csv(cache_path, index=False)
    return combined, cache_path


def load_predictor(config: type[Config], model_config: dict) -> KronosPredictor:
    tokenizer_kwargs = {}
    model_kwargs = {}
    if model_config.get("tokenizer_revision"):
        tokenizer_kwargs["revision"] = model_config["tokenizer_revision"]
    if model_config.get("model_revision"):
        model_kwargs["revision"] = model_config["model_revision"]

    tokenizer = KronosTokenizer.from_pretrained(model_config["tokenizer_name"], **tokenizer_kwargs)
    model = Kronos.from_pretrained(model_config["model_name"], **model_kwargs)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(
        model,
        tokenizer,
        device=config.device,
        max_context=model_config["max_context"],
    )


def dataframe_to_candles(df: pd.DataFrame) -> list[dict]:
    candles = []
    for _, row in df.iterrows():
        candles.append(
            {
                "timestamp": isoformat_timestamp(row["timestamps"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
                "amount": float(row.get("amount", 0.0)),
            }
        )
    return candles


def build_profile(config: type[Config], interval: str) -> dict:
    if interval not in config.interval_profiles:
        raise KeyError(f"Missing interval profile for {interval} in config.py")
    return dict(config.interval_profiles[interval])


def prepare_model_frame(df: pd.DataFrame, use_volume: bool) -> pd.DataFrame:
    if use_volume:
        return df[["open", "high", "low", "close", "volume", "amount"]].copy()
    return df[["open", "high", "low", "close"]].copy()


def compute_future_timestamps(last_timestamp: pd.Timestamp, interval: str, pred_len: int) -> pd.Series:
    step = interval_to_timedelta(interval)
    future_index = pd.date_range(
        start=last_timestamp + step,
        periods=pred_len,
        freq=step,
    )
    return pd.Series(future_index, name="timestamps")


def aggregate_paths(path_frames: list[pd.DataFrame]) -> dict:
    path_arrays = np.stack([frame[FEATURE_NAMES].to_numpy(dtype=np.float32) for frame in path_frames])
    timestamps = [isoformat_timestamp(ts) for ts in path_frames[0].index]
    mean_array = path_arrays.mean(axis=0)
    close_paths = path_arrays[:, :, CLOSE_INDEX]
    volume_paths = path_arrays[:, :, VOLUME_INDEX]
    close_quantiles = {
        "p10": np.quantile(close_paths, 0.10, axis=0).tolist(),
        "p25": np.quantile(close_paths, 0.25, axis=0).tolist(),
        "p50": np.quantile(close_paths, 0.50, axis=0).tolist(),
        "p75": np.quantile(close_paths, 0.75, axis=0).tolist(),
        "p90": np.quantile(close_paths, 0.90, axis=0).tolist(),
    }
    volume_quantiles = {
        "p10": np.quantile(volume_paths, 0.10, axis=0).tolist(),
        "p25": np.quantile(volume_paths, 0.25, axis=0).tolist(),
        "p50": np.quantile(volume_paths, 0.50, axis=0).tolist(),
        "p75": np.quantile(volume_paths, 0.75, axis=0).tolist(),
        "p90": np.quantile(volume_paths, 0.90, axis=0).tolist(),
    }
    mean_candles = []
    for index, timestamp in enumerate(timestamps):
        mean_candles.append(
            {
                "timestamp": timestamp,
                "open": float(mean_array[index, 0]),
                "high": float(mean_array[index, 1]),
                "low": float(mean_array[index, 2]),
                "close": float(mean_array[index, 3]),
                "volume": float(mean_array[index, 4]),
                "amount": float(mean_array[index, 5]),
            }
        )

    return {
        "timestamps": timestamps,
        "mean_candles": mean_candles,
        "close_paths": close_paths.tolist(),
        "close_quantiles": close_quantiles,
        "volume_paths": volume_paths.tolist(),
        "volume_quantiles": volume_quantiles,
        "path_count": int(path_arrays.shape[0]),
    }


def compute_summary(
    aggregated: dict,
    last_close: float,
) -> dict:
    close_paths = np.asarray(aggregated["close_paths"], dtype=np.float32)
    mean_close = np.mean(close_paths, axis=0)
    next_close_paths = close_paths[:, 0]
    final_close_paths = close_paths[:, -1]
    volume_paths = np.asarray(aggregated["volume_paths"], dtype=np.float32)
    mean_volume = np.mean(volume_paths, axis=0)
    next_volume_paths = volume_paths[:, 0]

    return {
        "sample_count": aggregated["path_count"],
        "next_close_mean": float(mean_close[0]),
        "next_close_median": float(np.median(next_close_paths)),
        "next_close_up_probability": float(np.mean(next_close_paths > last_close)),
        "horizon_end_up_probability": float(np.mean(final_close_paths > last_close)),
        "expected_return_pct_first_step": float(((mean_close[0] / last_close) - 1.0) * 100.0),
        "expected_return_pct_horizon_end": float(((mean_close[-1] / last_close) - 1.0) * 100.0),
        "next_volume_mean": float(mean_volume[0]),
        "next_volume_median": float(np.median(next_volume_paths)),
        "close_band_first_step": {
            "p10": float(aggregated["close_quantiles"]["p10"][0]),
            "p50": float(aggregated["close_quantiles"]["p50"][0]),
            "p90": float(aggregated["close_quantiles"]["p90"][0]),
        },
        "volume_band_first_step": {
            "p10": float(aggregated["volume_quantiles"]["p10"][0]),
            "p50": float(aggregated["volume_quantiles"]["p50"][0]),
            "p90": float(aggregated["volume_quantiles"]["p90"][0]),
        },
    }


def create_run_payload(
    config: type[Config],
    model_config: dict,
    symbol: str,
    interval: str,
    profile: dict,
    predictor: KronosPredictor,
    context_df: pd.DataFrame,
    kline_cache_path: Path,
    aggregated: dict,
    created_at: datetime,
) -> dict:
    last_timestamp = context_df["timestamps"].iloc[-1]
    last_close = float(context_df["close"].iloc[-1])
    run_id = (
        f"{symbol.lower()}_{interval}_ctx-{last_timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        f"_run-{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    )

    payload = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "symbol": symbol.upper(),
        "interval": interval,
        "data_source": {
            "exchange": "binance",
            "market": "spot",
            "kline_cache_path": str(kline_cache_path),
        },
        "model": {
            "model_name": model_config["model_name"],
            "tokenizer_name": model_config["tokenizer_name"],
            "model_revision": model_config.get("model_revision"),
            "tokenizer_revision": model_config.get("tokenizer_revision"),
            "device": predictor.device,
            "max_context": predictor.max_context,
        },
        "params": {
            "lookback": profile["lookback"],
            "pred_len": profile["pred_len"],
            "sample_count": profile["sample_count"],
            "temperature": profile["temperature"],
            "top_k": profile["top_k"],
            "top_p": profile["top_p"],
            "use_volume": profile["use_volume"],
        },
        "context": {
            "lookback_rows": int(len(context_df)),
            "history_start": isoformat_timestamp(context_df["timestamps"].iloc[0]),
            "history_end": isoformat_timestamp(last_timestamp),
            "last_close": last_close,
            "history": dataframe_to_candles(context_df),
        },
        "forecast": {
            "timestamps": aggregated["timestamps"],
            "mean_candles": aggregated["mean_candles"],
            "close_quantiles": aggregated["close_quantiles"],
            "close_paths": aggregated["close_paths"],
            "volume_quantiles": aggregated["volume_quantiles"],
            "volume_paths": aggregated["volume_paths"],
            "summary": compute_summary(aggregated, last_close),
        },
        "evaluation": {
            "status": "pending",
            "realized_steps": 0,
            "evaluated_at": None,
            "actual": [],
            "metrics": None,
        },
    }
    return payload


def save_run(cache_root: Path, run_payload: dict) -> Path:
    run_dir = ensure_dir(cache_root / "runs" / run_payload["symbol"] / run_payload["interval"])
    run_path = run_dir / f"{run_payload['run_id']}.json"
    with run_path.open("w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2)
    return run_path


def load_run(run_path: Path) -> dict:
    with run_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def contiguous_realized_prefix(run_payload: dict, kline_df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        pd.Timestamp(timestamp).tz_convert(UTC)
        if pd.Timestamp(timestamp).tzinfo is not None
        else pd.Timestamp(timestamp).tz_localize(UTC)
        for timestamp in run_payload["forecast"]["timestamps"]
    ]
    actual_map = {
        pd.Timestamp(row["timestamps"]).tz_convert(UTC): row
        for _, row in kline_df.iterrows()
    }

    realized_rows = []
    for timestamp in expected:
        row = actual_map.get(timestamp)
        if row is None:
            break
        realized_rows.append(row)

    if not realized_rows:
        return pd.DataFrame(columns=["timestamps"] + FEATURE_NAMES)
    realized_df = pd.DataFrame(realized_rows)
    realized_df.sort_values("timestamps", inplace=True)
    realized_df.reset_index(drop=True, inplace=True)
    return realized_df


def compute_evaluation(run_payload: dict, realized_df: pd.DataFrame) -> dict:
    realized_steps = len(realized_df)
    if realized_steps == 0:
        return run_payload["evaluation"]

    actual_close = realized_df["close"].to_numpy(dtype=np.float32)
    mean_close = np.array(
        [candle["close"] for candle in run_payload["forecast"]["mean_candles"][:realized_steps]],
        dtype=np.float32,
    )
    actual_volume = realized_df["volume"].to_numpy(dtype=np.float32)
    mean_volume = np.array(
        [candle["volume"] for candle in run_payload["forecast"]["mean_candles"][:realized_steps]],
        dtype=np.float32,
    )
    p25 = np.array(run_payload["forecast"]["close_quantiles"]["p25"][:realized_steps], dtype=np.float32)
    p75 = np.array(run_payload["forecast"]["close_quantiles"]["p75"][:realized_steps], dtype=np.float32)
    p10 = np.array(run_payload["forecast"]["close_quantiles"]["p10"][:realized_steps], dtype=np.float32)
    p90 = np.array(run_payload["forecast"]["close_quantiles"]["p90"][:realized_steps], dtype=np.float32)
    volume_p25 = np.array(
        run_payload["forecast"]["volume_quantiles"]["p25"][:realized_steps], dtype=np.float32
    )
    volume_p75 = np.array(
        run_payload["forecast"]["volume_quantiles"]["p75"][:realized_steps], dtype=np.float32
    )
    volume_p10 = np.array(
        run_payload["forecast"]["volume_quantiles"]["p10"][:realized_steps], dtype=np.float32
    )
    volume_p90 = np.array(
        run_payload["forecast"]["volume_quantiles"]["p90"][:realized_steps], dtype=np.float32
    )

    base_close = [run_payload["context"]["last_close"]]
    if realized_steps > 1:
        base_close.extend(actual_close[:-1].tolist())
    base_close = np.array(base_close, dtype=np.float32)

    predicted_direction = np.sign(mean_close - base_close)
    actual_direction = np.sign(actual_close - base_close)
    direction_accuracy = float(np.mean(predicted_direction == actual_direction))

    mse = float(np.mean((mean_close - actual_close) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(mean_close - actual_close)))
    volume_mse = float(np.mean((mean_volume - actual_volume) ** 2))
    volume_rmse = float(math.sqrt(volume_mse))
    volume_mae = float(np.mean(np.abs(mean_volume - actual_volume)))

    status = "complete"
    if realized_steps < run_payload["params"]["pred_len"]:
        status = "partial"

    return {
        "status": status,
        "realized_steps": realized_steps,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "actual": dataframe_to_candles(realized_df),
        "metrics": {
            "close_mae": mae,
            "close_rmse": rmse,
            "direction_accuracy": direction_accuracy,
            "coverage_p25_p75": float(np.mean((actual_close >= p25) & (actual_close <= p75))),
            "coverage_p10_p90": float(np.mean((actual_close >= p10) & (actual_close <= p90))),
            "volume_mae": volume_mae,
            "volume_rmse": volume_rmse,
            "volume_coverage_p25_p75": float(
                np.mean((actual_volume >= volume_p25) & (actual_volume <= volume_p75))
            ),
            "volume_coverage_p10_p90": float(
                np.mean((actual_volume >= volume_p10) & (actual_volume <= volume_p90))
            ),
        },
    }


def backfill_interval_runs(cache_root: Path, symbol: str, interval: str, kline_df: pd.DataFrame) -> list[Path]:
    run_dir = cache_root / "runs" / symbol.upper() / interval
    if not run_dir.exists():
        return []

    updated_paths = []
    for run_path in sorted(run_dir.glob("*.json")):
        run_payload = load_run(run_path)
        realized_df = contiguous_realized_prefix(run_payload, kline_df)
        evaluation = compute_evaluation(run_payload, realized_df)
        if evaluation != run_payload.get("evaluation"):
            run_payload["evaluation"] = evaluation
            with run_path.open("w", encoding="utf-8") as handle:
                json.dump(run_payload, handle, indent=2)
            updated_paths.append(run_path)
    return updated_paths


def build_run_summary(run_payload: dict) -> dict:
    metrics = run_payload["evaluation"]["metrics"] or {}
    summary = run_payload["forecast"]["summary"]
    return {
        "run_id": run_payload["run_id"],
        "created_at": run_payload["created_at"],
        "symbol": run_payload["symbol"],
        "interval": run_payload["interval"],
        "pred_len": run_payload["params"]["pred_len"],
        "sample_count": run_payload["params"]["sample_count"],
        "use_volume": run_payload["params"]["use_volume"],
        "forecast_start": run_payload["forecast"]["timestamps"][0],
        "forecast_end": run_payload["forecast"]["timestamps"][-1],
        "realized_steps": run_payload["evaluation"]["realized_steps"],
        "evaluation_status": run_payload["evaluation"]["status"],
        "next_close_mean": summary["next_close_mean"],
        "next_close_up_probability": summary["next_close_up_probability"],
        "horizon_end_up_probability": summary["horizon_end_up_probability"],
        "close_mae": metrics.get("close_mae"),
        "close_rmse": metrics.get("close_rmse"),
        "volume_mae": metrics.get("volume_mae"),
        "volume_rmse": metrics.get("volume_rmse"),
        "direction_accuracy": metrics.get("direction_accuracy"),
    }


def export_dashboard(cache_root: Path, website_data_root: Path, symbol: str) -> None:
    runs_root = cache_root / "runs" / symbol.upper()
    ensure_dir(website_data_root)
    website_runs_root = ensure_dir(website_data_root / "runs")

    dashboard = {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol": symbol.upper(),
        "intervals": {},
    }

    if not runs_root.exists():
        dashboard["message"] = "Run inference.py first to generate forecast data."
        with (website_data_root / "dashboard.json").open("w", encoding="utf-8") as handle:
            json.dump(dashboard, handle, indent=2)
        return

    for interval_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        interval = interval_dir.name
        runs = []
        for run_path in sorted(interval_dir.glob("*.json")):
            run_payload = load_run(run_path)
            runs.append(run_payload)

            destination = website_runs_root / run_path.name
            shutil.copyfile(run_path, destination)

        runs.sort(key=lambda run: run["created_at"], reverse=True)
        summaries = [build_run_summary(run) for run in runs]
        latest = summaries[0] if summaries else None
        dashboard["intervals"][interval] = {
            "latest_run_id": latest["run_id"] if latest else None,
            "latest_summary": latest,
            "runs": summaries,
        }

    with (website_data_root / "dashboard.json").open("w", encoding="utf-8") as handle:
        json.dump(dashboard, handle, indent=2)


def run_interval_forecast(
    config: type[Config],
    model_config: dict,
    predictor: KronosPredictor,
    symbol: str,
    interval: str,
) -> tuple[Path, dict]:
    profile = build_profile(config, interval)
    kline_df, kline_cache_path = update_kline_cache(
        config.cache_root, symbol, interval, config.binance_limit
    )
    if len(kline_df) < profile["lookback"]:
        raise RuntimeError(
            f"Not enough {interval} candles. Need {profile['lookback']}, got {len(kline_df)}."
        )

    context_df = kline_df.tail(profile["lookback"]).copy().reset_index(drop=True)
    future_timestamps = compute_future_timestamps(
        context_df["timestamps"].iloc[-1], interval, profile["pred_len"]
    )
    x_timestamp = context_df["timestamps"].reset_index(drop=True)
    model_df = prepare_model_frame(context_df, profile["use_volume"])

    with torch.no_grad():
        path_frames = predictor.predict_paths(
            df=model_df,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamps,
            pred_len=profile["pred_len"],
            T=profile["temperature"],
            top_k=profile["top_k"],
            top_p=profile["top_p"],
            sample_count=profile["sample_count"],
            verbose=False,
        )

    aggregated = aggregate_paths(path_frames)
    run_payload = create_run_payload(
        config=config,
        model_config=model_config,
        symbol=symbol,
        interval=interval,
        profile=profile,
        predictor=predictor,
        context_df=context_df,
        kline_cache_path=kline_cache_path,
        aggregated=aggregated,
        created_at=datetime.now(UTC),
    )
    run_path = save_run(config.cache_root, run_payload)
    backfill_interval_runs(config.cache_root, symbol, interval, kline_df)
    return run_path, run_payload


def print_run_summary(run_payload: dict) -> None:
    summary = run_payload["forecast"]["summary"]
    print(
        f"{run_payload['interval']} | next close mean: {summary['next_close_mean']:.2f} | "
        f"next volume mean: {summary['next_volume_mean']:.0f} | "
        f"next up prob: {summary['next_close_up_probability']:.1%} | "
        f"horizon up prob: {summary['horizon_end_up_probability']:.1%}"
    )


def main() -> int:
    model_config = dict(Config.inference_model)
    try:
        predictor = load_predictor(Config, model_config)
    except Exception as exc:
        print(f"Failed to load model/tokenizer: {exc}")
        return 1

    print(
        f"Loaded {model_config['model_name']} on {predictor.device} for symbol {Config.symbol.upper()}."
    )

    for interval in Config.interval_profiles:
        try:
            run_path, run_payload = run_interval_forecast(
                Config,
                model_config,
                predictor,
                Config.symbol,
                interval,
            )
            print(f"Saved run: {run_path}")
            print_run_summary(run_payload)
        except urllib.error.URLError as exc:
            print(f"Network error while fetching {interval} Binance data: {exc}")
            return 1
        except Exception as exc:
            print(f"Failed to run {interval} forecast: {exc}")
            return 1

    if Config.export_website_data:
        export_dashboard(
            cache_root=Config.cache_root,
            website_data_root=Config.website_data_root,
            symbol=Config.symbol,
        )
        print(f"Exported website data to {Config.website_data_root.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
