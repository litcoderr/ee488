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
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> Dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to fetch remote data") from exc

    data = {}
    for t in tickers:
        ticker = yf.Ticker(t)
        df = ticker.history(interval=interval, start=start_date, end=end_date, auto_adjust=auto_adjust)
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
    interval: str = "1d",
    auto_adjust: bool = False,
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
        return _load_from_yfinance(
            tickers, start_date, end_date, interval=interval, auto_adjust=auto_adjust
        )
    raise ValueError(f"Unknown data source '{source}'")


def download_yfinance_csv(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    data_dir: str,
    interval: str = "1d",
    auto_adjust: bool = False,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Fetch prices from yfinance and persist each ticker as ``<data_dir>/<ticker>.csv``."""
    cleaned = [t.strip().upper() for t in tickers if t and t.strip()]
    if not cleaned:
        raise ValueError("At least one ticker is required")

    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)

    data = _load_from_yfinance(
        cleaned, start_date=start_date, end_date=end_date, interval=interval, auto_adjust=auto_adjust
    )
    saved: Dict[str, Path] = {}
    for t, df in data.items():
        frame = df.copy()
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        frame.index.name = "Date"
        path = base / f"{t}.csv"
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}. Use overwrite=True to replace it.")
        frame.sort_index().to_csv(path)
        saved[t] = path
    return saved
