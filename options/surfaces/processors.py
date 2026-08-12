import numpy as np
import pandas as pd
import plotly.graph_objs as go

from collections import defaultdict
from threading import Lock, Event

from options.types.SSVIParams import SSVISurfParams, SSVITenorParams  # JL
from options.types.scope_pre_post import ScopePrePost, scoped_dates # jl
from options.types.dividend import get_dividends, Dividend # JL
from options.types.option_frame import OptionFrame # JL
from options.types.quote_side import QuoteSide # JL
from options.types.sym_date import SymDate #JL
from shared.modules.logger import info
from shared.paths import Paths
from typing import List, Iterable, Tuple, Dict
from datetime import timedelta, date, time, datetime
from dataclasses import dataclass
from itertools import chain, groupby
from multiprocessing import Pool, cpu_count
from plotly.subplots import make_subplots

from options.frame_builder import get_option_frame
from options.helper import cache_to_disk, find_loc_every_x_pc, timer, exclude_outlier_quotes
from options.types.calibration_item import df2calibration_items, CalibrationItem
from options.types.enums import Resolution #JL
from options.types.equity import Equity #JL
from options.types.iv_surface_essvi import IVSurface, MetricSSVI
from shared.plotting import show


def get_ivs_mp(sym_dates: List[SymDate], scope: ScopePrePost | str = ScopePrePost.all, arb_free=False, mp=False) -> List[IVSurface]:
    payloads = []
    for sym_date in sym_dates:
        # if not check_data_presence(sym, take):
        #     continue
        sym = sym_date.symbol.lower()
        release_date = sym_date.date
        equity = Equity(sym)
        resolution = Resolution.second
        seq_ret_threshold = 0.002

        dates = scoped_dates(release_date, scope)
        payloads += [(equity, dates, resolution, seq_ret_threshold, arb_free)]

    n_processes = min(cpu_count() // 2, len(payloads)) if mp else 1
    info(f'get_ivs_mp: Processing {len(payloads)} payloads in {n_processes}.')
    if mp:
        with Pool(n_processes) as pool:
            res = pool.starmap(get_v_ivs, payloads)
    else:
        res = [ivs for p in payloads if (ivs := get_v_ivs(*p))]
    res = [x for x in res if x]

    return [x for x in list(chain(*res))]


def scope_df(x: pd.DataFrame, start_ts=None, end_ts=None, moneyness_fit = 'moneyness_fwd_ln', min_mny = -0.3, max_mny = 0.3):
    df = x.copy()
    df = df[(df[moneyness_fit] < max_mny) & (df[moneyness_fit] > min_mny)]
    if start_ts:
        df = df[df.index.get_level_values('ts') >= start_ts]
    if end_ts:
        df = df[df.index.get_level_values('ts') < end_ts]
    return df


def get_timestamp_slices(timestamps, slice_duration=30 * 60 * 1000):
    slices = []
    for i in range(len(timestamps)):
        slice_start = timestamps[i]
        slice_end = slice_start + slice_duration
        slice_timestamps = [t for t in timestamps if slice_start <= t < slice_end]
        slices.append(slice_timestamps)
    return slices


def get_time_buckets_start_end(
        start_dt: date,
        end_dt: date,
        min_time: time = time(9, 30),
        max_time: time = time(16, 0),
        bucket_size: timedelta = timedelta(minutes=30)
):
    """buckets beyond 1 day wouldn't be supported by the downstream funcs. can only have 1 calc date."""
    for start_ts in pd.date_range(start_dt, end_dt + timedelta(days=1), freq=bucket_size):
        end_ts = start_ts + bucket_size
        if min_time <= start_ts.time() <= max_time and end_ts.time() <= max_time:
            # info(start_ts, end_ts)
            yield start_ts, end_ts


def get_time_buckets_by_equity_returns(
        ps_spot,
        seq_ret_threshold: float = 0.002,
        first_snap_time: time = time(9, 32),
        min_period=timedelta(seconds=30)
):
    v_loc = find_loc_every_x_pc(ps_spot, seq_ret_threshold)
    yielded = False
    for i, end_ts in enumerate(v_loc):
        if i == 0:
            continue

        if not yielded:
            start_ts = v_loc[0]
        else:
            start_ts = v_loc[i - 1]

        if (end_ts.time() < first_snap_time or
                end_ts - start_ts < min_period):
            continue

        yielded = True
        yield start_ts, end_ts


def get_time_buckets_start_end_by_sample_count(v_ts: pd.DatetimeIndex, buckets_per_day: int = 20) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    """introduces look-ahead bias. rather pick some parameter or mix of them."""
    v_ts_dates = pd.to_datetime(v_ts).date
    samples_per_day = {dt: len(list(grouper)) for dt, grouper in groupby(v_ts_dates)}
    for dt, n_samples in samples_per_day.items():
        bucket_size = n_samples // buckets_per_day
        for i in range(0, n_samples - bucket_size, bucket_size):
            start_ts = v_ts[v_ts_dates == dt][i]

            is_last_bucket = i + bucket_size > n_samples - bucket_size / 2
            if is_last_bucket:
                end_ts = v_ts[v_ts_dates == dt][-1]
            else:
                end_ts = v_ts[v_ts_dates == dt][i + bucket_size]
            # info(start_ts, end_ts)
            yield start_ts, end_ts

@timer
def df_qt2surfaces(
    equity: Equity,
    df_q: pd.DataFrame,
    df_t: pd.DataFrame,
    df_equity,
    side: QuoteSide | str = QuoteSide.mid,
    arb_free=False,
    n_workers = 8,
    seq_ret_threshold_surface=None
) -> List[IVSurface]:
    payloads = []
    if seq_ret_threshold_surface:
        for start, end in get_time_buckets_by_equity_returns(df_equity["close"], seq_ret_threshold=seq_ret_threshold_surface):
            payloads.append((equity, scope_df(df_q, start, end), scope_df(df_t, start, end), start.date(), side, arb_free))
    else:
        start = datetime(1970, 1, 1)
        end = datetime(2099, 1, 1)
        calc_date = df_q.index.get_level_values('ts')[0].date()
        payloads.append((equity, scope_df(df_q, start, end), scope_df(df_t, start, end), calc_date, side, arb_free))

    if not payloads:
        return []

    info(f'df_qt2surfaces(): Processing {len(payloads)} slices of IV surface for {equity} - {side}')

    res: List[IVSurface | None] = []
    if len(payloads) == 1:
        res = [df_qt2surface(*payloads[0])]
    else:
        res = run_with_rolling_warmstart(df_qt2surface, payloads, res, n_workers)

    return [x for x in res if x is not None]

def run_with_rolling_warmstart(func, payloads, res, n_workers):
    """
    func: callable like df_qt2surface(..., x0_ivs=...)
    payloads: iterable of tuples (all args except x0_ivs)
    res: list of completed results (seeded already if you want)
    """
    it = iter(payloads)
    lock = Lock()
    done_event = Event()

    pending = 0
    exhausted = False
    first_exc = None

    def maybe_finish():
        # called with lock held
        if exhausted and pending == 0:
            done_event.set()

    def submit_next(pool):
        nonlocal pending, exhausted, first_exc

        with lock:
            if first_exc is not None:
                return
            try:
                payload = next(it)
            except StopIteration:
                exhausted = True
                maybe_finish()
                return

            x0_snapshot = list(res)  # snapshot at submission time
            pending += 1

        pool.apply_async(
            func,
            args=(*payload, x0_snapshot),
            callback=lambda ivs: on_ok(ivs, pool),
            error_callback=lambda exc: on_err(exc, pool),
        )

    def on_ok(ivs, pool):
        nonlocal pending
        with lock:
            pending -= 1
            if ivs is not None:
                res.append(ivs)
            maybe_finish()
        # submit after updating res (so next job sees newer res)
        info(f'run_with_rolling_warmstart(): pending calibrations: {pending}')
        submit_next(pool)

    def on_err(exc, pool):
        nonlocal pending, first_exc
        with lock:
            pending -= 1
            if first_exc is None:
                first_exc = exc
            done_event.set()

    pool = Pool(n_workers)
    try:
        for _ in range(n_workers):
            submit_next(pool)

        done_event.wait()

        if first_exc is not None:
            raise first_exc
    finally:
        pool.terminate()  # safest if exceptions happened
        pool.join()

    return res

def v_ivs_to_weighted_metrics(v_ivs: List[IVSurface]) -> SSVISurfParams:
    theta = defaultdict(float)
    rho = defaultdict(float)
    psi = defaultdict(float)
    total_sample = 0
    for ivs in v_ivs:
        for tenor, ci in ivs.calibration_items.items():
            ivs.params = SSVISurfParams({k: SSVITenorParams(*v) if isinstance(v, tuple) else v for k, v in ivs.params.items()})
            theta[tenor] += ivs.params[tenor].theta * len(ci.price)
            rho[tenor] += ivs.params[tenor].rho * len(ci.price)
            psi[tenor] += ivs.params[tenor].psi * len(ci.price)
            total_sample += len(ci.price)
    inst = SSVISurfParams()
    for dt in theta.keys():
        inst[dt] = SSVITenorParams(theta[dt]/total_sample, rho[dt]/total_sample, psi[dt]/total_sample)
    return inst

def v_ivs_to_x0(v_ivs: List[IVSurface]|Tuple) -> Dict[date, List[float]]:
    out = defaultdict(list)
    count = defaultdict(int)
    for ivs in v_ivs:
        for tenor, ci in ivs.calibration_items.items():
            if len(ci.price) > count[tenor]:
                out[tenor] = list(ivs.params[tenor])
                count[tenor] = len(ci.price)
    return out

def get_calibration_items(calc_date: date, df_q: pd.DataFrame, dividends: List[Dividend], side: QuoteSide|str, df_t: pd.DataFrame = None) -> List[CalibrationItem]:
    if side == QuoteSide.mid:
        # calibration_items_fill = df2calibration_items(df_t, calc_date, iv_col_nm='fill_iv', price_col_nm='close', spot_col_nm='spot', rf=rate, dividends=dividends,
        #                                               weight_col_nm=None, vega_col_nm='vega_fill_iv')
        calibration_items_quotes = df2calibration_items(df_q, calc_date, iv_col_nm='mid_price_iv', price_col_nm='mid_price', spot_col_nm='spot', dividends=dividends,
                                                        weight_col_nm=None)
        calibration_items = calibration_items_quotes
        # Weight them somewhat making fills equal to quotes
        # calibration_items = union_calibration_items(calibration_items_quotes, calibration_items_fill)

    elif side == QuoteSide.bid:
        calibration_items = df2calibration_items(df_q, calc_date, iv_col_nm='bid_iv', price_col_nm='bid_close', spot_col_nm='spot', dividends=dividends,
                                                 weight_col_nm=None)
    elif side == QuoteSide.ask:
        calibration_items = df2calibration_items(df_q, calc_date, iv_col_nm='ask_iv', price_col_nm='ask_close', spot_col_nm='spot', dividends=dividends,
                                                 weight_col_nm=None)
    else:
        raise ValueError(f'Unknown side: {side}')

    if not calibration_items:
        return None

    return calibration_items


@timer
def df_qt2surface(equity: Equity, df_q: pd.DataFrame, df_t: pd.DataFrame, calc_date: date, side: QuoteSide | str = QuoteSide.mid, arb_free=False,
                  x0_ivs: List[IVSurface]|Tuple = (), n_samples_per_tenor=1000) -> IVSurface | None:
    sym = equity.symbol.lower()
    dividends = get_dividends(sym, start=calc_date)

    ivs = IVSurface(equity)
    calibration_items = get_calibration_items(calc_date, df_q, dividends, side, df_t)
    if not calibration_items:
        return None

    # Fit entire surface
    samples_all = {item.tenor_dt: item for item in calibration_items}
    samples = {item.tenor_dt: item.downsample(min(n_samples_per_tenor, len(item.price))) for item in calibration_items}
    n_samples_all = sum([len(ci.price) for ci in samples_all.values()])
    n_samples = sum([len(ci.price) for ci in samples.values()])
    info(f'df_qt2surface(): Downsampled from {n_samples_all} to {n_samples}')
    # if arb_free:
    #     ivs.calibrate_surface_arb_free(samples)
    # else:
    # Pick IVS with largest sample size as X0

    x0 = v_ivs_to_x0(x0_ivs)
    ivs.calibrate_surface(equity, samples, calc_date, x0_in=x0)

    # ivs.plot_params_rmse(release_date)
    # calibration_items_bid = df2calibration_items(scope_df(df_q), release_date, iv_col_nm='bid_iv', price_col_nm='bid_close', spot_col_nm='spot', rf=rate, dividends=dividends, weight_col_nm='vega_mid_iv')
    # calibration_items_ask = df2calibration_items(scope_df(df_q), release_date, iv_col_nm='ask_iv', price_col_nm='ask_close', spot_col_nm='spot', rf=rate, dividends=dividends, weight_col_nm='vega_mid_iv')
    # ivs_0.plot_all_prices(release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)
    # ivs_0.plot_all_smiles(release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)
    # ivs.plot_surface()

    return ivs


def get_v_ivs(equity: Equity, dates: List[date], resolution: Resolution|str, seq_ret_threshold: float, arb_free=False, seq_ret_threshold_surface=None) -> List[IVSurface]:
    v_ivs = []
    for dt in dates:
        v_ivs += get_ivs_by_date(equity, [dt], resolution, seq_ret_threshold, QuoteSide.mid, arb_free, seq_ret_threshold_surface)
    return v_ivs


def get_df_qt(option_frame: OptionFrame, equity: Equity, dates: List[date]):
    df = option_frame.df_options.sort_index()
    df_trades = option_frame.df_option_trades.sort_index()

    # A little cleaning
    df_trades = exclude_outlier_quotes(df_trades, [], equity)

    df_q = get_df_xy(df, dates)
    df_t = get_df_xy(df_trades, dates)
    return df_q, df_t


@cache_to_disk('v_ivs', Paths.path_analysis_frames)
def get_ivs_by_date(equity: Equity, dates: List[date], resolution: Resolution, seq_ret_threshold: float, side=QuoteSide.mid, arb_free=False, seq_ret_threshold_surface=None) -> List[IVSurface]:
    info(f'get_ivs_by_date(): {equity} - {",".join((d.isoformat() for d in dates))}')
    option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
    df_q, df_t = get_df_qt(option_frame, equity, dates)
    v_ivs = df_qt2surfaces(equity, df_q, df_t, option_frame.df_equity, side, arb_free=arb_free, seq_ret_threshold_surface=seq_ret_threshold_surface)
    return v_ivs


class DistributionNormal:
    def __init__(self, mu, sd, cnt):
        self.mu = mu
        self.sd = sd
        self.cnt = cnt

    def confidence_interval(self, z=1.96, cnt=None):
        # x̄ ± z* σ/√n
        return self.mu + z * self.sd / (cnt or self.cnt) ** 0.5


@dataclass
class IVSurfaceSSVIMetricDistribution:
    underlying: str
    # right: OptionRight | str
    metric: MetricSSVI
    tenor_dt: date
    ts_start: datetime
    ts_end: datetime
    distribution: DistributionNormal


def get_ssvi_distributions(v_ivs: Iterable[IVSurface]) -> List[IVSurfaceSSVIMetricDistribution]:
    df = pd.DataFrame(chain(*[ivs.datapoints for ivs in v_ivs]))

    results = []

    # can combine into single statement?
    for (underlying, metric, tenor_dt), df_gp in df.groupby(['underlying', 'model_param', 'tenor_dt']):
        for ts_start, gp in df_gp[['ts_end', 'value']].groupby(df_gp['ts_end'].dt.date):
            mean = gp.mean()['value']
            sd = gp.std()['value']
            cnt = gp.count()['value']
            dist = DistributionNormal(mean, sd, cnt)

            results.append(IVSurfaceSSVIMetricDistribution(underlying, metric, tenor_dt, ts_start, ts_start, dist))

    return results


def plot_ssvi_params_dist_across_dates(v_ivs: Iterable[IVSurface], fn=None, open_browser=True, z=1.96):
    """use get_ssvi_distributions"""
    if not v_ivs:
        return
    dists = get_ssvi_distributions(v_ivs)

    # rights = [OptionRight.call, OptionRight.put]
    metrics = MetricSSVI.names()
    # subs = list(chain(*[[f'{right} {metric}' for right in rights] for metric in metrics]))
    subs = metrics

    for underlying, gp_dists in groupby(dists, lambda x: x.underlying):
        fig = make_subplots(rows=4, cols=1, subplot_titles=subs)
        for (tenor_dt, metric), gp2_dists in groupby(gp_dists, lambda x: (x.tenor_dt, x.metric)):
            gp2_dists = sorted(gp2_dists, key=lambda x: x.ts_start)
            ix_row = metrics.index(metric) + 1

            x = [d.ts_start for d in gp2_dists]
            v_mu = [d.distribution.mu for d in gp2_dists]
            v_sd_up = [d.distribution.confidence_interval(z) for d in gp2_dists]
            v_sd_dn = [d.distribution.confidence_interval(-z) for d in gp2_dists]

            fig.add_trace(go.Scatter(x=x, y=v_mu, mode='markers', marker=dict(size=4), name=f'{tenor_dt} {metric}'), row=ix_row, col=1)
            fig.add_trace(go.Scatter(x=x, y=v_sd_up, mode='markers', marker=dict(size=4), name=f'{tenor_dt} {metric} SD up'), row=ix_row, col=1)
            fig.add_trace(go.Scatter(x=x, y=v_sd_dn, mode='markers', marker=dict(size=4), name=f'{tenor_dt} {metric} SD dn'), row=ix_row, col=1)

            fig.update_layout(title=f'{underlying} SSVI params over time', autosize=True)
        show(fig, fn=fn or f'essvi_{underlying}_params_over_time.html', open_browser=open_browser)


def get_df_xy(df: pd.DataFrame, dates: List[date]):
    """Calibrating based on prices, excluding by IV is not good"""

    if all((c in df.columns for c in ('bid_iv', 'ask_iv'))):
        df = df[(df['bid_iv'] < 4) & (df['ask_iv'] < 4)]
    elif all((c in df.columns for c in ('fill_iv', ))):
        df = df[(df['fill_iv'] < 4)]
    elif all((c in df.columns for c in ('bid_close', 'close', 'ask_close'))):
        df = df[(df['bid_close'] <= df['close']) | (df['ask_close'] >= df['close'])]

    # Scope by time of day
    v_ts = df.index.get_level_values('ts').unique()
    v_ts_scoped = v_ts[np.isin(v_ts.date, np.array(dates)) & (v_ts.hour >= 10) & (v_ts.hour < 16)]

    return df.loc[v_ts_scoped]


def remove_outdated_cache():
    """
    Load every option_frame from cache
    if create date is older than last_modified date of dependent files, delete it.

    Also delete any v_ivs based on that frame...
    """
    import os, pickle
    for root, dirs, fns in os.walk(Paths.path_analysis_frames):
        for i, fn in enumerate(fns):
            if i % 500 == 0:
                print(i)
            if fn.startswith('option_frame-'):
                p = os.path.join(root, fn)
                try:
                    with open(p, 'rb') as f:
                        obj: OptionFrame = pickle.load(f)
                except ModuleNotFoundError:
                    print(f'Error Removed {p}')
                    os.remove(p)
                    continue
                # get option data file
                dir_ = Paths.path_data.joinpath('option\\usa\\second', obj.equity.symbol.lower())
                fp = dir_.joinpath(f'{obj.start.strftime('%Y%m%d')}_quote_american.zip')
                if not os.path.exists(fp):
                    continue
                mtime_src = os.path.getmtime(fp)
                mtime_frame = os.path.getmtime(p)
                if mtime_src > mtime_frame:
                    print(f'Removed {p}')
                    os.remove(p)

def remove_invalid_v_ivs(tag='v_ivs'):
    import os, pickle
    from options.helper import get_pkl_cache_key
    for root, dirs, fns in os.walk(Paths.path_analysis_frames):
        for i, fn in enumerate(fns):
            if i % 500 == 0:
                print(i)
            if fn.startswith(tag):
                p = os.path.join(root, fn)
                try:
                    with open(p, 'rb') as f:
                        obj: List[IVSurface] = pickle.load(f)
                except ModuleNotFoundError:
                    print(f'Error Removed {p}')
                    os.remove(p)
                    continue
                if len(obj) == 0:
                    print(f'No surfaced Removed {p}')
                    os.remove(p)
                    continue
                ivs = obj[0]
                fn = get_pkl_cache_key('option_frame', *(ivs.underlying, [ivs.last_calibration_ts().date()], (), (), Resolution.second,  0.002))
                fp = os.path.join(root, fn)
                if not os.path.exists(fp):
                    print(f'Removed, no frame: {p}')
                    os.remove(p)

if __name__ == '__main__':
    # remove_outdated_cache()
    equity = Equity("ORCL")
    dt = date(2026, 3, 9)
    resolution = Resolution.second
    seq_ret_threshold=0.002
    print(get_ivs_by_date(equity, [dt], resolution, seq_ret_threshold, QuoteSide.mid, False, None))
