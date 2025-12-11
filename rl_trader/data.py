import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return pd.date_range(start=start, end=end, freq="B")


def _generate_synthetic_series(
    tickers: Iterable[str], start_date: str, end_date: str, rng: np.random.Generator
) -> Dict[str, pd.DataFrame]:
    dates = _date_range(start_date, end_date)
    data = {}
    for t in tickers:
        base = rng.normal(loc=0.001, scale=0.02, size=len(dates)).cumsum()
        price = 100 * np.exp(base)
        high = price * (1 + rng.uniform(0, 0.02, size=len(dates)))
        low = price * (1 - rng.uniform(0, 0.02, size=len(dates)))
        open_price = price * (1 + rng.uniform(-0.01, 0.01, size=len(dates)))
        close_price = price
        volume = rng.integers(low=1_000_000, high=5_000_000, size=len(dates))
        df = pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close_price,
                "Volume": volume,
            },
            index=dates,
        )
        data[t] = df
    return data


def _load_from_csv(
    tickers: Iterable[str], data_dir: str, start_date: str, end_date: str
) -> Dict[str, pd.DataFrame]:
    base = Path(data_dir)
    data = {}
    for t in tickers:
        path = base / f"{t}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for ticker {t}: {path}")
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        df = df.loc[start_date:end_date]
        data[t] = df
    return data


def _load_from_yfinance(
    tickers: Iterable[str], start_date: str, end_date: str
) -> Dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to fetch remote data") from exc

    data = {}
    for t in tickers:
        ticker = yf.Ticker(t)
        df = ticker.history(interval="1d", start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            raise RuntimeError(f"No data returned from yfinance for {t}")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        data[t] = df
    return data


def load_price_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    source: str = "yfinance",
    data_dir: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, pd.DataFrame]:
    if rng is None:
        rng = np.random.default_rng()

    source = source.lower()
    if source == "synthetic":
        return _generate_synthetic_series(tickers, start_date, end_date, rng)
    if source == "csv":
        if data_dir is None:
            raise ValueError("data_dir is required when source='csv'")
        return _load_from_csv(tickers, data_dir, start_date, end_date)
    if source == "yfinance":
        return _load_from_yfinance(tickers, start_date, end_date)
    raise ValueError(f"Unknown data source '{source}'")
