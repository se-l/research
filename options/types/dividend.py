import os
from functools import lru_cache

import pandas as pd
import yfinance as yf

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Union, Optional, List

from shared.modules.logger import info, error
from shared.paths import Paths, mkdir

NO_DIVIDEDNS = {'PANW', 'ONON', 'XPO', 'PATH', 'CDNS', 'BIDU', 'CBRE', 'CDNS', 'CRWD', 'DLTR', 'DOCU', 'KMX', 'LI', 'MDB',
                'MRNA', 'NEOG', 'NFLX', 'ON', 'ONON', 'PANW', 'PATH', 'PDD', 'PLTR', 'SNOW', 'WDAY', 'XPO', 'ZK', "ESLT",
                "PDD"}


@dataclass
class Dividend:
    ticker: str
    ex_date: date
    amount: float
    predicted: bool = False


def _dividends_dir() -> Path:
    return Path(Paths.path_data).joinpath('equity', 'usa', 'dividends')


def _dividends_csv_path(ticker: str) -> Path:
    return _dividends_dir().joinpath(f"{ticker.upper()}.csv")


def _to_yyyymmdd(d: date) -> str:
    return d.strftime('%Y%m%d')


def _from_yyyymmdd(s: str) -> date:
    return datetime.strptime(s, '%Y%m%d').date()


def _write_dividends_csv(ticker: str, dividends: List[Dividend]) -> None:
    # Ensure directory exists
    mkdir(_dividends_dir())
    df = pd.DataFrame([
        {
            'ExDate': _to_yyyymmdd(d.ex_date),
            'Amount': float(d.amount),
            'Ccy': 'USD'
        }
        for d in sorted(dividends, key=lambda x: x.ex_date)
    ])
    # Even if empty, write headers to establish the file
    csv_path = _dividends_csv_path(ticker)
    df.to_csv(csv_path, index=False)


def _read_dividends_csv(ticker: str) -> Optional[List[Dividend]]:
    path = _dividends_csv_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype={'ExDate': str})
    except Exception:
        # If reading fails, treat as absent
        return None
    if df.empty:
        # present but empty
        return []
    out: List[Dividend] = []
    for _, row in df.iterrows():
        try:
            exd = _from_yyyymmdd(str(row['ExDate']))
            amt = float(row['Amount'])
            out.append(Dividend(ticker=ticker.upper(), ex_date=exd, amount=amt))
        except Exception:
            continue
    return out


def load_from_disk(ticker: str, start: Union[str, date, datetime], end: Union[str, date, datetime]) -> Optional[List[Dividend]]:
    # Normalize dates
    if isinstance(start, str):
        start = datetime.strptime(start, '%Y-%m-%d').date()
    if isinstance(end, str):
        end = datetime.strptime(end, '%Y-%m-%d').date()
    data = _read_dividends_csv(ticker)
    if data is None:
        return None  # signal not found on disk
    # Filter range
    return [d for d in data if start <= d.ex_date <= end]


def _fetch_yahoo_all(ticker: str) -> List[Dividend]:
    try:
        info(f'Fetching Dividends from Yahoo for {ticker}')
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends is None or len(dividends) == 0:
            return []
        if isinstance(dividends, pd.Series):
            dividends = dividends.reset_index()
            dividends.columns = ['Date', 'Amount']
        dividends['Ticker'] = ticker.upper()
        dividends = dividends[['Date', 'Amount', 'Ticker']]
        recs = dividends.rename(columns={'Date': 'ex_date', 'Amount': 'amount', 'Ticker': 'ticker'})
        # Convert to records and ensure date objects
        out: List[Dividend] = []
        for r in recs.to_dict('records'):
            exd = r['ex_date'].date() if hasattr(r['ex_date'], 'date') else r['ex_date']
            if isinstance(exd, datetime):
                exd = exd.date()
            out.append(Dividend(ticker=r['ticker'], ex_date=exd, amount=float(r['amount'])))
        return out
    except Exception:
        error(f'Failed to fetch Dividends from Yahoo for {ticker}')
        return []


def _infer_frequency_months(ex_dates: List[date]) -> int:
    # Use recent 8-12 intervals if available
    if len(ex_dates) < 2:
        return 3  # default quarterly when unknown
    ex_dates_sorted = sorted(ex_dates)
    deltas = [(ex_dates_sorted[i] - ex_dates_sorted[i-1]).days for i in range(1, len(ex_dates_sorted))]
    if not deltas:
        return 3
    median_days = sorted(deltas)[len(deltas)//2]
    # Map median days to months
    candidates = {
        1: 30,
        2: 60,
        3: 91,
        4: 121,
        6: 182,
        12: 365
    }
    # Choose closest
    months = min(candidates.keys(), key=lambda m: abs(candidates[m] - median_days))
    return months


def _add_months(d: date, months: int) -> date:
    # Simple month addition handling year wrap and month-end
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # Find last day of target month by moving to first of next month then subtract 1 day
    if m == 12:
        next_y, next_m = y + 1, 1
    else:
        next_y, next_m = y, m + 1
    first_next_month = date(next_y, next_m, 1)
    last_day_prev = first_next_month - timedelta(days=1)
    day = min(d.day, last_day_prev.day)
    return date(y, m, day)


def _forecast_dividends(ticker: str, history: List[Dividend], until: date) -> List[Dividend]:
    if not history:
        return []
    # Only consider history up to today for forecasting cadence and last amount
    today = datetime.now().date()
    hist = [d for d in history if d.ex_date <= today]
    if not hist:
        return []
    hist_sorted = sorted(hist, key=lambda d: d.ex_date)
    last = hist_sorted[-1]
    months = _infer_frequency_months([d.ex_date for d in hist_sorted[-12:]])
    forecast: List[Dividend] = []
    next_date = _add_months(last.ex_date, months)
    # Cap forecast horizon to 2 years from today
    horizon = min(until, today + timedelta(days=int(365.25*2)))
    while next_date <= horizon:
        # Annualized growth 5% based on exact time delta in years from last.ex_date
        years = (next_date - last.ex_date).days / 365.25
        amt = last.amount * ((1.0 + 0.05) ** years)
        forecast.append(Dividend(ticker=ticker.upper(), ex_date=next_date, amount=float(amt), predicted=True))
        next_date = _add_months(next_date, months)
    return forecast


@lru_cache(maxsize=None)
def get_dividends(ticker: str, start: Union[str, date, datetime], end: Union[str, date, datetime]=None) -> List[Dividend]:
    if ticker.upper() in NO_DIVIDEDNS:  # might need to confirm once in a while...
        return []

    end = end or start + timedelta(days=365*3)
    # Normalize dates
    if isinstance(start, str):
        start_date = datetime.strptime(start, '%Y-%m-%d').date()
    elif isinstance(start, datetime):
        start_date = start.date()
    else:
        start_date = start
    if isinstance(end, str):
        end_date = datetime.strptime(end, '%Y-%m-%d').date()
    elif isinstance(end, datetime):
        end_date = end.date()
    else:
        end_date = end

    # Try load from disk (do not fetch if present per requirement)
    loaded = load_from_disk(ticker, start_date, end_date)
    if loaded is None:
        # Not on disk: fetch from Yahoo (all history), save, then proceed
        all_hist = _fetch_yahoo_all(ticker)
        _write_dividends_csv(ticker, all_hist)
        history = all_hist
    else:
        # File exists; also read full file to enable forecasting
        history = _read_dividends_csv(ticker) or []

    # Combine history within range and add forecasts up to end_date
    today = datetime.now().date()
    in_range_hist = [d for d in history if start_date <= d.ex_date <= min(end_date, today)]
    forecasts = _forecast_dividends(ticker, history, end_date)
    in_range_forecasts = [d for d in forecasts if start_date <= d.ex_date <= end_date]
    combined = sorted(in_range_hist + in_range_forecasts, key=lambda d: d.ex_date)
    return combined


def fetch_dividends(
        ticker: str,
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        source: str = 'yahoo'
) -> List[Dividend]:
    """
    Fetch dividend data for a given ticker symbol and date range.

    Parameters:
    -----------
    ticker : str
        The ticker symbol of the stock (e.g., 'AAPL', 'MSFT')

    start_date : str, date, or datetime
        The start date for dividend data retrieval
        String format should be 'YYYY-MM-DD'

    end_date : str, date, or datetime, optional
        The end date for dividend data retrieval
        String format should be 'YYYY-MM-DD'
        If None, defaults to current date

    source : str, optional
        The data source to use. Currently supports:
        - 'yahoo': Yahoo Finance (default)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing dividend information with columns:
        - Date: Dividend ex-date
        - Amount: Dividend amount
        - Ticker: Ticker symbol

    Raises:
    -------
    ValueError
        If an unsupported source is specified
    ImportError
        If required libraries aren't installed
    """
    # Standardize dates
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str) and end_date is not None:
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Default end_date to today if not provided
    if end_date is None:
        end_date = datetime.now().date()

    # Validate source
    if source.lower() != 'yahoo':
        raise ValueError(f"Source '{source}' not supported. Currently only 'yahoo' is supported.")

    # Fetch from Yahoo Finance
    try:
        # Get ticker data
        stock = yf.Ticker(ticker)

        # Get dividend data
        dividends = stock.dividends

        if dividends.empty:
            print(f"No dividends found for {ticker} in the specified period.")
            return []

        # Convert to DataFrame if it's a Series
        if isinstance(dividends, pd.Series):
            dividends = dividends.reset_index()
            dividends.columns = ['Date', 'Amount']

        # Filter by date range
        dividends = dividends[
            (dividends['Date'].dt.date >= start_date) &
            (dividends['Date'].dt.date <= end_date)
            ]

        # Add ticker column
        dividends['Ticker'] = ticker

        # Reorder columns
        dividends = dividends[['Date', 'Amount', 'Ticker']]

        return [Dividend(**kw) for kw in dividends.rename(columns={'Date': 'ex_date', 'Amount': 'amount', 'Ticker': 'ticker'}).to_dict('records')]

    except ImportError:
        raise ImportError("This function requires the 'yfinance' package. Install it with 'pip install yfinance'")

    except Exception as e:
        print(f"Error fetching dividends for {ticker}: {str(e)}")
        return []

def search_for_zero_divs():
    no_divs = []
    for ticker in os.listdir(r'D:\trade\data\option\usa\second'):
        ticker = ticker.upper()
        if not get_dividends(ticker, '2022-01-01', (datetime.now() + timedelta(days=int(365.25 * 2))).strftime('%Y-%m-%d')):
            print(ticker)
            no_divs.append(ticker)
    print(no_divs)
    return no_divs


# Example usage:
if __name__ == "__main__":
    # Example: get dividends (including forecast) for next 2 years
    search_for_zero_divs()
    # divs = get_dividends('PANW', '2023-01-01', (datetime.now() + timedelta(days=int(365.25*2))).strftime('%Y-%m-%d'))
    # for d in divs:
    #     print(d)