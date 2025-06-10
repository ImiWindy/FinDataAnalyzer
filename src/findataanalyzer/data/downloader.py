"""
Module for downloading and saving historical market data using yfinance.
"""
import logging
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

def download_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    timeframes: Dict[str, str],
    data_dir: Path = Path("data")
):
    """
    Downloads historical data for given symbols and timeframes and saves it to CSV files.

    Args:
        symbols: A list of ticker symbols to download (e.g., ['DIA', 'GC=F']).
        start_date: The start date for the data in 'YYYY-MM-DD' format.
        end_date: The end date for the data in 'YYYY-MM-DD' format.
        timeframes: A dictionary mapping desired timeframe name to yfinance interval string
                    (e.g., {'1h': '60m', '15m': '15m'}).
        data_dir: The directory where the data files will be saved.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting data download process...")

    for symbol in symbols:
        for tf_name, yf_interval in timeframes.items():
            file_path = data_dir / f"{symbol}_{tf_name}.csv"
            
            # Check if data already exists to avoid re-downloading
            if file_path.exists():
                logger.info("Data for %s (%s) already exists. Skipping.", symbol, tf_name)
                continue

            try:
                logger.info("Downloading data for %s, timeframe: %s...", symbol, tf_name)
                # Note: yfinance has limitations on intraday data lookback periods.
                # For '15m' and '60m', it's usually limited to the last 730 days.
                data = yf.download(
                    tickers=symbol,
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    auto_adjust=True, # Automatically adjusts for splits and dividends
                    progress=False
                )

                if data.empty:
                    logger.warning("No data returned for %s with interval %s.", symbol, yf_interval)
                    continue
                
                # Standardize column names to lowercase
                data.columns = [col.lower() for col in data.columns]
                
                data.to_csv(file_path)
                logger.info("Successfully downloaded and saved data to %s", file_path)

            except Exception as e:
                logger.error("Failed to download data for %s (%s): %s", symbol, tf_name, e)

if __name__ == '__main__':
    # Example usage when running the script directly
    from datetime import datetime, timedelta

    # Download data for the past year
    end = datetime.now()
    start = end - timedelta(days=365)
    
    SYMBOLS_TO_DOWNLOAD = ['DIA', 'GC=F'] # Dow Jones ETF, Gold Futures
    TIMEFRAMES_TO_DOWNLOAD = {
        '1h': '60m',
        '15m': '15m',
        '5m': '5m' # Using 5m as the trigger timeframe
    }

    download_data(
        symbols=SYMBOLS_TO_DOWNLOAD,
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
        timeframes=TIMEFRAMES_TO_DOWNLOAD
    ) 