import argparse
import datetime as dt

from rl_trader.data import download_yfinance_csv


def parse_tickers(raw: str):
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def main():
    parser = argparse.ArgumentParser(description="Download yfinance data and save per-ticker CSV files.")
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker symbols, e.g., AAPL,MSFT,GOOG")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data", help="Directory to save CSV files")
    parser.add_argument("--interval", default="1d", help="yfinance interval such as 1d or 1h")
    parser.add_argument("--auto-adjust", action="store_true", help="Download adjusted prices")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs")
    args = parser.parse_args()

    tickers = parse_tickers(args.tickers)
    saved = download_yfinance_csv(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        interval=args.interval,
        auto_adjust=args.auto_adjust,
        overwrite=args.overwrite,
    )
    print("Saved CSV files:")
    for t, path in saved.items():
        print(f"  {t}: {path}")


if __name__ == "__main__":
    main()
