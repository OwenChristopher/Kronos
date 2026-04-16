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
        "evaluation_days": 60,
        "pred_len": 1,
        "use_monte_carlo_average": True,
        "monte_carlo_paths": 100,
        "temperature": 0.6,
        "top_k": 0,
        "top_p": 0.90,
        "use_volume": True,
        "apply_fees": True,
        "entry_fee_rate": 0.0004,
        "exit_fee_rate": 0.0003,
        # Position-holding strategy: only enter on high conviction, hold through noise
        "entry_threshold": 0.003,       # predicted |return| must exceed 0.3% to enter
        "exit_threshold": 0.001,        # close position when conviction drops below 0.1%
        "confidence_threshold": 0.60,   # MC path agreement required to enter (0.5 = no filter)
        "min_hold_bars": 3,             # hold at least 3 bars (12h for 4h candles) before allowing exit
    }
