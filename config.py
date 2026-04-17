from pathlib import Path


class Config:
    symbol = "BTCUSDT"

    device = None
    cache_root = Path("forecast_cache")
    website_data_root = Path("website/public/data")
    binance_limit = 1000
    export_website_data = True

    inference_model = {
        "model_name": "NeoQuasar/Kronos-base",
        "tokenizer_name": "NeoQuasar/Kronos-Tokenizer-base",
        "model_revision": None,
        "tokenizer_revision": None,
        "max_context": 512,
    }

    backtest_model = {
        "model_name": "NeoQuasar/Kronos-base",
        "tokenizer_name": "NeoQuasar/Kronos-Tokenizer-base",
        "model_revision": None,
        "tokenizer_revision": None,
        "max_context": 512,
    }

    # Set use_volume to False if you want to mirror the paper's crypto setup.
    interval_profiles = {
        "15m": {
            "lookback": 360,
            "pred_len": 16,
            "sample_count": 50,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.95,
            "use_volume": True,
        },
        "1h": {
            "lookback": 240,
            "pred_len": 12,
            "sample_count": 50,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.95,
            "use_volume": True,
        },
    }

    backtest = {
        "symbol": "BTCUSDT",
        "interval": "4h",
        "lookback": 180,
        "evaluation_days": 30,
        "use_monte_carlo_average": True,
        "monte_carlo_paths": 100,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.95,
        "use_volume": True,
        "apply_fees": True,
        "entry_fee_rate": 0.0004,
        "exit_fee_rate": 0.0003,
        # Fee-aware threshold (multipliers applied to round-trip fee):
        #   enter only when |predicted_return| > entry_threshold_multiplier * rt_fee
        #   stay in position while opposite signal < exit_threshold_multiplier * rt_fee
        # Round-trip fee at default rates = 0.07% -> entry ~0.14%, exit ~0.035%.
        "entry_threshold_multiplier": 2.0,
        "exit_threshold_multiplier": 0.5,
    }
