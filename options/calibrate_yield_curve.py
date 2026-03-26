import numpy as np

from itertools import chain
from collections import defaultdict
from copy import copy
from datetime import date
from typing import List, Dict, Tuple
from scipy.optimize import least_squares

from options.frame_builder import get_option_frame, available_sym_dates
from options.surfaces.processors import get_df_qt, get_calibration_items, scoped_dates
from options.helper import get_moneyness_fwd, date_to_sod
from options.types.calibration_item import CalibrationItem
from options.types.dividend import get_dividends
from options.types.enums import Resolution, OptionRight
from options.types.equity import Equity
from options.types.iv_surface_essvi import IVSurface, f_essvi_total_variance, logistic_spread_weighted_residuals
from options.types.option import get_price_cuda
from options.types.quote_side import QuoteSide
from options.types.scope_pre_post import ScopePrePost
from shared.modules.logger import info
from shared.yield_curve import YieldCurve, ZeroCurveData


def get_weighted_params(v_ivs: List[IVSurface]) -> Dict[date, List[float]]:
    total = defaultdict(int)
    weighted_params: Dict[date, List[float]] = {}

    for ivs in v_ivs:
        for tenor, params in ivs.params.items():
            theta, rho, psi = params
            n = len(ivs.calibration_items[tenor].bid_price)
            total[tenor] += n
            if not weighted_params.get(tenor):
                weighted_params[tenor] = [0, 0, 0]
            weighted_params[tenor][0] += theta * n
            weighted_params[tenor][1] += rho * n
            weighted_params[tenor][2] += psi * n

    for tenor in weighted_params.keys():
        weighted_params[tenor][0] /= total[tenor]
        weighted_params[tenor][1] /= total[tenor]
        weighted_params[tenor][2] /= total[tenor]

    return weighted_params


def f_minimize_put_call_surface_difference(
        rate: np.ndarray[float],
        time_step: int,
        zero_rates: ZeroCurveData,
        equity: Equity,
        params: Dict[date, Tuple[float, ...]],
        s,
        k,
        t,
        bids,
        asks,
        v_expiry,
        v_is_call,
        dividends,
        calculation_date
) -> float:
    adjusted_zero_rates = copy(zero_rates)
    adjusted_zero_rates.rates[time_step] = float(rate[0])

    v_mny_fwd_ln = np.log(get_moneyness_fwd(
        equity,
        strike=k,
        spot=s,
        tenor=t,
        t0=date_to_sod(calculation_date),
        yield_curve_in = adjusted_zero_rates
    ))

    # Construct v_iv. need to get unique tenors and index
    v_iv = np.empty_like(s)
    tenors_unique, inv_index = np.unique(v_expiry, return_inverse=True)
    for i, tenor_dt in enumerate(tenors_unique):
        theta, rho, psi = params[tenor_dt]
        ix_t = inv_index == i
        n_t = sum(ix_t)
        v_tenor = t[ix_t]
        v_iv[ix_t] = np.sqrt(f_essvi_total_variance(v_mny_fwd_ln[ix_t], np.array([theta]*n_t), np.array([rho]*n_t), np.array([psi]*n_t)) / v_tenor)

    model_prices = get_price_cuda(s=s, k=k, t=t,
        v_is_call=v_is_call,
        iv=v_iv,
        dividends=dividends,
        calculation_date=calculation_date,
        yield_curve=adjusted_zero_rates
    )
    residuals, _, _, _ = logistic_spread_weighted_residuals(model_prices, bids, asks)
    return residuals

def calibrate_yield_curve(
        equity: Equity,
        samples: Dict[date, CalibrationItem],
        calculation_date: date,
        weighted_params,
        dividends,
        min_max_mny_fwd_ln = (-0.2, 0.2),
        delta_rates_bound=0.1,
        verbose=1
) -> ZeroCurveData:
    zero_rates = YieldCurve().get_zero_curve(calculation_date)  # Not passing equity, get the uncalibrated curve...

    vals = samples.values()
    bids = np.array(list(chain(*[i.bid_price for i in vals])))
    asks = np.array(list(chain(*[i.ask_price for i in vals])))
    s = np.array(list(chain(*[i.spot for i in vals])))
    k = np.array(list(chain(*[i.strike for i in vals])))
    t = np.array(list(chain(*[[i.tenor] * len(i.strike) for i in vals])))
    v_expiry = np.array(list(chain(*[[i.tenor_dt] * len(i.strike) for i in vals])))
    v_is_call = np.array(list(chain(*[[1 if r == OptionRight.call else 0 for r in i.rights] for i in vals])))

    v_mny_fwd_ln = np.log(get_moneyness_fwd(
        equity,
        strike=k,
        spot=s,
        tenor=t,
        t0=date_to_sod(calculation_date),
        yield_curve_in=zero_rates
    ))
    ix_scoped = (min_max_mny_fwd_ln[0] < v_mny_fwd_ln) & (v_mny_fwd_ln < min_max_mny_fwd_ln[1])

    for i, (tenor, rate) in enumerate(zip(zero_rates.times, zero_rates.rates)):
        info(f'calibrate_yield_curve(): Fitting {calculation_date} {tenor}')
        fit_res = least_squares(
            fun=f_minimize_put_call_surface_difference,
            x0=rate,
            args=(
                i,
                zero_rates,
                equity,
                weighted_params,
                s[ix_scoped],
                k[ix_scoped],
                t[ix_scoped],
                bids[ix_scoped],
                asks[ix_scoped],
                v_expiry[ix_scoped],
                v_is_call[ix_scoped],
                dividends,
                calculation_date
            ),
            verbose=verbose,
            max_nfev=5_000,
            bounds=([rate - delta_rates_bound], [rate + delta_rates_bound])
        )

        zero_rates.rates[i] = float(fit_res.x[0])

    return zero_rates

def calibrate_yield_curve_and_store(
        v_ivs: List[IVSurface],
        calc_date: date,
        equity: Equity,
        resolution: Resolution = Resolution.second,
        side = QuoteSide.mid,
        seq_ret_threshold=0.002,
        plot=False,
        verbose=1,
) -> ZeroCurveData:
    dividends = get_dividends(equity.symbol, calc_date)
    if plot:
        YieldCurve.plot(YieldCurve().get_zero_curve(calc_date))
    params = get_weighted_params(v_ivs)
    option_frame = get_option_frame(equity, [calc_date], resolution=resolution, seq_ret_threshold=seq_ret_threshold)
    df_q, df_t = get_df_qt(option_frame, equity, [calc_date])
    calibration_items = get_calibration_items(calc_date, df_q, dividends, side, df_t)
    if calibration_items is None:
        info(f'calibrate_yield_curve(): No calibration items found')
        return
    samples = {item.tenor_dt: item for item in calibration_items if item.tenor_dt in params}
    recalibrated_curve = calibrate_yield_curve(equity, samples, calculation_date=calc_date, weighted_params=params, dividends=dividends, verbose=verbose)
    if plot:
        YieldCurve.plot(recalibrated_curve)
    YieldCurve.store_calibrated_rates(equity, calc_date, recalibrated_curve)
    return recalibrated_curve


def run(tickers: List[str]):
    tickers_in = ','.join(tickers).upper()
    resolution = Resolution.second
    seq_ret_threshold = 0.002
    side = 'mid'

    sym_dates = available_sym_dates(tickers_in)
    for sym_date in sym_dates:
        sym = sym_date.symbol.lower()
        equity = Equity(sym)
        release_date = sym_date.date

        for dt in scoped_dates(release_date, ScopePrePost.all):
            calc_date = dt
            calibrate_yield_curve_and_store(calc_date, equity, resolution, side, seq_ret_threshold=seq_ret_threshold)


if __name__ == '__main__':
    """
     Given rolling 3 days worth of IVS surface quotes,
     return a zero rate curve that minimizes the price difference between
     put mid and call mid curves.
     Calibrate piecewise, 1 tenor at the time and vary all the rates up to that tenor, fix them,
     move onto next tenor. Risk of having some weird curves if more than 2 rates are calibrated simultaneously.
     Ideally, reward smoothness somehow, simply minimizing the difference in rates would do, a term smaller than price difference.

     The calibrated curve is for a symbol as of a date. Store as such in
     DATA alternative interest-rate usa daily {sym}-{YYYYMMDD}.csv
     Schema
     Tenor|Rate

     The calibrated curve is loaded in YieldCurve when a symbol is provided, defaults to non-calibrated curve.

     Re-evaluate, same IV, but pricer will pick up different rates curve and arrive at better put-call parity.
    """
    run(['FDX'])

