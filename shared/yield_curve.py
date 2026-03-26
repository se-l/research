import os
import csv

import pandas as pd
import polars as pl

from datetime import datetime, date, timedelta
from functools import lru_cache
from typing import Dict, List, Tuple
from dataclasses import dataclass

from options.types.equity import Equity
from shared.constants import dt_fmt_ymd
from shared.paths import Paths

TENOR = 'Tenor'
RATE = 'Rate'


@dataclass
class ZeroCurveData:
    times: List[float]
    rates: List[float]

    def __copy__(self):
        return ZeroCurveData(self.times, self.rates)


class YieldCurve:
    _instance = None
    _initialized = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YieldCurve, cls).__new__(cls)
            # Maps date (datetime.date) to a dict of {tenor_in_years: zero_rate}
            cls._instance._bill_rates = {}
            cls._instance._par_yields = {}
            cls._instance._zero_curves = {}
        return cls._instance

    def get_file_path(self, market: str, curve_type: str, year: int) -> str:
        return os.path.join(Paths.path_data_interest_rate, market, curve_type, f"{year}.csv")

    def load_bill_rates(self, market: str, dt: date):
        path = self.get_file_path(market, "daily-treasury-bill-rates", dt.year)
        if not os.path.exists(path):
            return

        # Indices for Coupon Equivalent (CE) columns: 2, 4, 6, 8, 10, 12, 14
        ce_indices = [2, 4, 6, 8, 10, 12, 14]
        tenors = [4.0 / 52, 6.0 / 52, 8.0 / 52, 13.0 / 52, 17.0 / 52, 26.0 / 52, 52.0 / 52]

        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                try:
                    row_date = datetime.strptime(row[0], "%m/%d/%Y").date()

                    if row_date not in self._bill_rates:
                        self._bill_rates[row_date] = {}

                    for i, idx in enumerate(ce_indices):
                        if idx < len(row) and row[idx]:
                            rate = float(row[idx]) / 100.0
                            self._bill_rates[row_date][tenors[i]] = rate
                except (ValueError, IndexError):
                    continue

    def load_par_yield_curve(self, market: str, dt: date):
        path = self.get_file_path(market, "daily-treasury-par-yield-curve-rates", dt.year)
        if not os.path.exists(path):
            return

        tenors = [1.0 / 12, 1.5 / 12, 2.0 / 12, 3.0 / 12, 4.0 / 12, 6.0 / 12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                try:
                    row_date = datetime.strptime(row[0], "%m/%d/%Y").date()
                    par_yields = []

                    for i, tenor in enumerate(tenors):
                        if (i + 1) < len(row) and row[i + 1]:
                            rate = float(row[i + 1]) / 100.0
                            par_yields.append((tenor, rate))

                    if par_yields:
                        bootstrapped = self.bootstrap_to_zero_rates(par_yields)
                        if row_date not in self._par_yields:
                            self._par_yields[row_date] = {}

                        self._par_yields[row_date].update(bootstrapped)
                except (ValueError, IndexError):
                    continue

    def initialize(self, dt: date, market: str):
        if dt not in self._initialized:
            self._initialized.add(dt)
            self.load_bill_rates(market, dt)
            self.load_par_yield_curve(market, dt)

    @staticmethod
    def has_calibrated_curve(calculation_date: date, equity: Equity = None, market="usa") -> bool:
        if equity is None:
            return False
        root = Paths.path_data_interest_rate / market / 'daily'
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'{equity.symbol.upper()}-{calculation_date.strftime(dt_fmt_ymd)}.csv'
        return path.exists()

    @staticmethod
    def store_calibrated_rates(equity: Equity, calculation_date: date, curve: ZeroCurveData, market="usa"):
        root = Paths.path_data_interest_rate / market / 'daily'
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'{equity.symbol.upper()}-{calculation_date.strftime(dt_fmt_ymd)}.csv'
        df = pl.DataFrame(zip(curve.times, curve.rates), schema=[TENOR, RATE]).cast(
            {TENOR: pl.Float64, RATE: pl.Float64}
        )
        df.write_csv(path)

    @staticmethod
    def load_calibrated_rates(equity: Equity, calculation_date: date, market="usa") -> ZeroCurveData:
        root = Paths.path_data_interest_rate / market / 'daily'
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'{equity.symbol.upper()}-{calculation_date.strftime(dt_fmt_ymd)}.csv'
        if not path.exists():
            raise ValueError(f'load_calibrated_rates(): {path} does not exist.')
        df = pl.read_csv(path)
        return ZeroCurveData(times=list(df.get_column(TENOR)), rates=list(df.get_column(RATE)))

    def get_last_zero_curve(self, calculation_date: date, equity: Equity = None, delta_days=-14, market="usa") -> ZeroCurveData:
        for pd_dt in pd.date_range(calculation_date + timedelta(days=delta_days), calculation_date)[::-1]:
            if self.has_calibrated_curve(pd_dt, equity, market):
                return self.get_zero_curve(pd_dt, equity, market=market)
        return self.get_zero_curve(calculation_date, market=market)

    @lru_cache()
    def get_zero_curve(self, calculation_date: date, equity: Equity = None, market="usa") -> ZeroCurveData:
        if self.has_calibrated_curve(calculation_date, equity, market):
            return self.load_calibrated_rates(equity, calculation_date, market)

        if calculation_date not in self._initialized:
            self.initialize(calculation_date, market)

        if calculation_date not in self._zero_curves:
            bills = self._bill_rates.get(calculation_date, {})
            pars = self._par_yields.get(calculation_date, {})

            if not bills and not pars:
                return ZeroCurveData(times=[], rates=[])

            # Start with bills as zero rates (usually <= 1 year)
            combined_zero_rates = {t: r for t, r in bills.items()}

            combined_zero_rates.update({t: r for t, r in pars.items() if t > max(bills.keys(), default=0)})

            sorted_tenors = sorted(combined_zero_rates.keys())
            self._zero_curves[calculation_date] = {t: combined_zero_rates[t] for t in sorted_tenors}

        curve = self._zero_curves[calculation_date]
        sorted_tenors = sorted(curve.keys())

        return ZeroCurveData(
            times=[float(t) for t in sorted_tenors],
            rates=[float(curve[t]) for t in sorted_tenors]
        )

    def bootstrap_to_zero_rates(self, par_yields: List[Tuple[float, float]]) -> Dict[float, float]:
        zero_rates = {}
        sorted_par = sorted(par_yields, key=lambda x: x[0])

        for t, c in sorted_par:
            if t <= 1.0:
                zero_rates[t] = c
            else:
                # Treasury bonds pay semi-annually. We need PV of all coupons at 0.5, 1.0, 1.5 ... t
                coupon = c / 2.0
                pv_coupons = 0.0

                # Number of semi-annual payments
                num_payments = int(round(t * 2))
                for i in range(1, num_payments):
                    payment_time = i * 0.5
                    # Interpolate the zero rate for the payment time
                    z_i = self.interpolate_zero_rate(zero_rates, payment_time)
                    pv_coupons += coupon / ((1 + z_i / 2.0) ** (2 * payment_time))

                remaining_price = 1.0 - pv_coupons
                if remaining_price <= 0:
                    zero_rates[t] = c
                    continue

                z_n = 2.0 * (((1.0 + coupon) / remaining_price) ** (1.0 / (2.0 * t)) - 1.0)
                zero_rates[t] = z_n

        return zero_rates

    def interpolate_zero_rate(self, zero_rates: Dict[float, float], t: float) -> float:
        """Linear interpolation of zero rates."""
        times = sorted(zero_rates.keys())
        if t <= times[0]:
            return zero_rates[times[0]]
        if t >= times[-1]:
            return zero_rates[times[-1]]

        # Find surrounding points
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                t1, t2 = times[i], times[i + 1]
                r1, r2 = zero_rates[t1], zero_rates[t2]
                return r1 + (r2 - r1) * (t - t1) / (t2 - t1)
        return zero_rates[times[-1]]

    @staticmethod
    def plot(curve: ZeroCurveData):
        import matplotlib.pyplot as plt
        plt.plot(curve.times, curve.rates)
        plt.xlabel('Time (years)')
        plt.ylabel('Zero Rate')
        plt.title('Zero Curve')
        plt.show()

if __name__ == "__main__":
    print(YieldCurve().get_zero_curve(date(2025, 12, 24)))
