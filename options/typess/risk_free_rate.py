from typing import Dict, List

import pandas as pd
import numpy as np
import yfinance as yf
import polars as pl

from pathlib import Path
from datetime import timedelta, date
from scipy.interpolate import interp1d
from sympy import false

from shared.modules.logger import info
from shared.paths import Paths, mkdir

IR = 'interest-rate'
ix2tenor = {
    # "^SOFR": 0,  # Example: Overnight rate
    "^IRX": 0.25,  # 3-month Treasury yield
    # "^IRY": 1,  # Replace with 1-year yield symbol
    # "^TZY": 2,  # Replace with 2-year yield symbol
    "^FVX": 5,  # 5-year Treasury yield
    "^TNX": 10,  # 10-year Treasury yield
    "^TYX": 30  # 30-year Treasury yield
}

def get_fp(dt: date) -> Path:
    return Paths.path_data_alternative.joinpath(IR, 'usa', f"risk_free_rate_{dt}.csv")

def fetch_and_store_risk_free_rate(dt: date, symbol = "^IRX") -> bool:
    """
    Fetch the risk-free rates for a given date and save them to a CSV file.
    # Proxy for the risk-free rate (3-month Treasury Bill (^IRX) on Yahoo Finance)
    """
    info(f"Fetching risk-free rate for {symbol} {dt}...")
    df = yf.download(symbol, start=dt, end=dt + timedelta(days=1), interval="1d")

    if df.empty:
        info(f"No df found for {dt}.")
        return False

    # Adjust or process raw df (convert the close price to a percentage if needed)
    fp = get_fp(dt)
    mkdir(fp.parent)
    (df["Close"] / 100).rename(columns={symbol: 'Rate'}).to_csv(fp)
    info(f"Data stored in {fp}")
    return True


def get_standard_terms():
    """
    Define standard market terms for interpolation (e.g., 1 month, 3 months, 6 months, 1 year, etc.).
    """
    return np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 10, 20, 30])  # Terms in years


def interpolate_risk_free_rate(rates, terms, calculation_terms):
    """
    Interpolate the risk-free rates for standard terms.

    :param rates: List or array of observed risk-free rates.
    :param terms: List or array of terms (in years) corresponding to the observed rates.
    :param calculation_terms: List of terms (in years) for which the interpolated rates are needed.
    :returns: Interpolated rates for the given calculation terms.
    """
    # Ensure the terms and rates are sorted
    sorted_indices = np.argsort(terms)
    terms = np.array(terms)[sorted_indices]
    rates = np.array(rates)[sorted_indices]

    # Perform interpolation with cubic spline
    interpolator = interp1d(terms, rates, kind="cubic", fill_value="extrapolate")

    # Calculate interpolated rates
    interpolated_rates = interpolator(calculation_terms)
    return interpolated_rates


def fetch_and_store_multiple_rf_rates(dt: date, symbols: List[str]) -> bool:
    """
    Fetch and store risk-free rates for multiple terms (e.g., SOFR, 3M, 6M, 1Y... 30Y).

    :param dt: The date for which to fetch rates.
    :param symbols: A dictionary mapping term lengths (months/years) to Yahoo Finance symbols.
                    Example: {"SOFR": "^SOFR", "3M": "^IRX", "10Y": "^TNX"}.
    :return: True if data is successfully fetched, False otherwise.
    """
    info(f"Fetching risk-free rates for {dt}...")

    all_rates = {}
    # Fetch data for each symbol
    df = yf.download(symbols, start=dt, end=dt + timedelta(days=1), interval="1d")

    if df.empty:
        info(f"No data found on {dt}.")
        return false

    # Convert Close price to decimal (basis points to rate)
    df_rates = pd.DataFrame(df.loc[dt.isoformat(), ["Close"]])
    df_rates.index = df_rates.index.get_level_values('Ticker')
    df_rates = df_rates.transpose()[symbols]

    # Save all rates to a CSV file (with terms as the header)
    fp = get_fp(dt)
    mkdir(fp.parent)

    # Preparing the dataframe with terms as headers
    df = pd.DataFrame([all_rates])
    df.index = [dt]
    df_rates.to_csv(fp)
    info(f"Data stored in {fp}")

    return True


def get_rf_rate(dt: date) -> Dict[str, float]:
    """
    Load risk-free rates from disk for a specific date and return as a Polars Series.

    :param dt: The date for which to load the risk-free rate.
    :return: Polars Series containing risk-free rates.
    """
    fp = get_fp(dt)
    available_dt = dt
    if not fp.exists():
        for i in range(10):
            if fetch_and_store_multiple_rf_rates(dt - timedelta(days=i), symbols=list(ix2tenor.keys())):
                available_dt = dt - timedelta(days=i)
                break

    fp = get_fp(available_dt)
    info(f"Loading risk-free rate data from {fp}")
    df = pd.read_csv(fp)
    df.set_index(df.columns[0], inplace=True)
    return df.loc[available_dt.isoformat()].to_dict()


def get_rf_curve(t0: date):
    """
    Interpolate a risk-free rate curve from a specific date for the next 3 years.

    :param t0: The date for which to calculate the interpolated curve.
    :return: A Pandas DataFrame with interpolated daily rates for the next 3 years.
    """
    fp = get_fp(t0)
    if not fp.exists():
        raise FileNotFoundError(f"No risk-free rate file found for {t0}")

    rates = get_rf_rate(t0)
    values = [v for v in rates.values() if v and not np.isnan(v)]
    terms = [ix2tenor.get(k) for k, v in rates.items() if v and not np.isnan(v)]

    # Generate future terms (every day for the next 3 years)
    future_dates = pd.date_range(t0, t0 + timedelta(days=365 * 3), freq="D")
    calculation_terms = np.arange(1 / 365, 3 + 1 / 365, 1 / 365)  # 1 day to 3 years

    # Interpolate rates
    interpolated_rates = interpolate_risk_free_rate(values, terms, calculation_terms)

    # Construct the curve
    curve_df = pd.DataFrame({
        "Date": future_dates[:-1],
        "Rate": interpolated_rates
    })
    info(f"Interpolated rate curve for {t0}: \n{curve_df.head()}")
    return curve_df


if __name__ == "__main__":
    dt = date(2025, 8, 22)

    # Load risk-free rate data for dt
    rf_rate = get_rf_rate(dt)
    info(f"Risk-free rates for {dt}: \n{rf_rate}")

    # Generate interpolated curve for the next 3 years
    rf_curve = get_rf_curve(dt)
    info(f"Interpolated curve saved to {rf_curve}")
