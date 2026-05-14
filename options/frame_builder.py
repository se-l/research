import math
import os.path
import pandas as pd
import numpy as np

from importlib import reload
from functools import partial
from typing import List
from itertools import chain
from datetime import date, time, datetime

from options.types.scope_pre_post import ScopePrePost, scoped_dates
from options.types.dividend import get_dividends
from options.types.option import get_vega_cuda, get_delta_cuda
from options.types.sym_date import SymDate
from shared.modules.logger import info, error
from shared.paths import Paths
from shared.constants import DiscountRateMarket, EarningsPreSessionDates, USA, SECOND
from options.helper import find_loc_every_x_pc, year_quarter, earnings_download_dates_start_end, spot_from_df_equity_into_options, df2atm_iv, \
    enrich_atm_iv_by_right, \
    ps2intrinsic_value, quotes2multi_index_df, load_option_trades, join_spot, join_quotes, ps2mid_iv_if_nonzero, get_dividend_yield, \
    get_moneyness_fwd, cache_to_disk, get_v_tenor_from_index, df2iv
from options.types.enums import SecurityType, TickType, Resolution, OptionRight
from options.types.equity import Equity
from options.client import Client
from options.types.option_contract import OptionContract
from options.types.option_frame import OptionFrame
import options.client as mClient
from shared.utils.decorators import time_it

reload(mClient)
client = mClient.Client()


def unpack_mi_df_index(ps: pd.Series) -> tuple:
    ts = ps.name[0]
    expiry = ps.name[1]
    strike = ps.name[2]
    right = ps.name[3]
    return ts, expiry, strike, right


def ps2mid_iv(ps: pd.Series) -> pd.Series:
    return (ps['ask_iv'] + ps['bid_iv']) / 2


class ATMHelper:
    def __init__(self, expiry: date, strike: float, calculation_date: date, net_yield: float):
        self.expiry = expiry
        self.strike = strike
        self.calculation_date = calculation_date
        self.net_yield = net_yield

    def PV_K(self):
        # PV(K) = K * exp(-(r - q) * (T - t))
        # net_yield = rate - dividend_yield = r - q
        # T-t = DTE/365
        return self.strike * np.exp(-self.net_yield * (self.expiry - self.calculation_date).days / 365)


def generate_option_frame(start, end, equity, resolution, seq_ret_threshold, version='1', keep_seconds: List[datetime] = None):
    client = Client()
    trades_eq = client.history([equity], start, end, resolution, TickType.trade, SecurityType.equity)
    if str(equity) not in trades_eq:
        error(f'No equity trades for {equity} in {start} - {end}.')

    ps_spot = trades_eq[str(equity)]['close']
    # removing non-RTH hours from spot because not present in quotes
    ps_spot = ps_spot[(ps_spot.index.time >= time(9, 30)) & (ps_spot.index.time <= time(16, 0))]

    # refactor to simply loading all of them...
    contracts = client.central_volatility_contracts(equity, start, end, n=3000)
    contracts = list(chain(*contracts.values()))
    print(f'# Contracts loaded: {len(contracts)}')

    quotes = client.history(contracts, start, end, resolution, TickType.quote, SecurityType.option)
    quotes = {OptionContract.from_contract_nm(k): v for k, v in quotes.items()}
    if not quotes:
        error(f'No option quotes for {equity} in {start} - {end}.')

    trades = load_option_trades(contracts, start, end, resolution)
    if not trades:
        error(f'No option trades for {equity} in {start} - {end}.')
        df_trades = None
    else:
        trades = join_spot(trades, ps_spot)
        trades = join_quotes(trades, quotes)
        df_trades = quotes2multi_index_df(trades)

    loc = find_loc_every_x_pc(ps_spot, seq_ret_threshold)
    if keep_seconds:
        loc = list(sorted(list(set(loc).union({pd.Timestamp(i) for i in keep_seconds}))))
        ps_spot = pd.concat((pd.Series(None, index=keep_seconds, name=ps_spot.name), ps_spot))
        ps_spot = ps_spot[~ps_spot.index.duplicated()].sort_index().ffill()
        assert ps_spot.isna().sum() == 0

    # FFill every loc that's not within a quote df
    for c, df in quotes.items():
        dft = df.merge(pd.DataFrame(index=loc), how='outer', left_index=True, right_index=True).sort_index().ffill()
        quotes[c] = dft.loc[loc]

    ps_spot = ps_spot.loc[loc]
    df = quotes2multi_index_df(quotes)

    # Dropping expired derivatives - check the speed of this... check by ts.date() not every single ts...
    ix_drop = []
    for ts, sub_df in df.groupby(level='ts'):
        for expiry, sub_sub_df in sub_df.groupby(level='expiry'):
            if expiry < ts.date():
                ix_drop.extend(sub_sub_df.index)
    df.drop(ix_drop, inplace=True)

    # Store
    option_frame = OptionFrame(
        equity=equity,
        df_options=df,
        df_equity=pd.DataFrame(ps_spot),
        df_option_trades=df_trades,
        start=start,
        end=end,
        resolution=resolution,
        seq_ret_threshold=seq_ret_threshold,
        version=version,
        # version='23Q4',
        ts_created=datetime.now()
    )
    # option_frame.store()
    return option_frame


def enrich_quotes(underlying: Equity, option_frame):
    df = option_frame.df_options
    spot_from_df_equity_into_options(option_frame)
    df['tenor'] = get_v_tenor_from_index(df)
    df['strike_flt'] = df.index.get_level_values('strike').astype(float)
    df['intrinsic_value'] = df.apply(partial(ps2intrinsic_value, spot_column='spot'), axis=1)
    df['date'] = df.index.get_level_values('ts').date

    v_calc_date = np.array(list(map(lambda x: x.date(), df.index.get_level_values('ts').to_pydatetime())))
    assert len(set(v_calc_date)) == 1, 'The current dividends function only allows a single calculation date'
    dividends = get_dividends(underlying.symbol.upper(), start=v_calc_date[0])

    df['bid_iv'] = df2iv(df, price_col_nm='bid_close', dividends=dividends, calculation_date=df['date'].values)
    df['ask_iv'] = df2iv(df, price_col_nm='ask_close', dividends=dividends, calculation_date=df['date'].values)
    df['mid_iv'] = df.apply(ps2mid_iv_if_nonzero, axis=1)
    df['extrinsic_bid'] = df['bid_close'] - df['intrinsic_value']
    df['extrinsic_ask'] = df['ask_close'] - df['intrinsic_value']

    # df['vega_mid_iv'] = 100 * get_vega(df['spot'].values, df['strike_flt'].values, df['tenor'].values, df['mid_iv'].values, rate, div_amounts, div_times)
    # df['vega_mid_iv'] = df.apply(partial(ps2vega, iv_col='mid_iv', calendar=calendar, day_count=day_count, dividend=dividend), axis=1)
    # (abs(df['vega_mid_iv'] / 100 - v_vega) < 0.0001).sum() == len(df)

    df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
    df['mid_price_iv'] = df2iv(df, price_col_nm='mid_price', dividends=dividends, calculation_date=df['date'].values, equity=underlying)

    df['bid_delta'] = None
    df['ask_delta'] = None
    df['delta'] = None

    v_is_call = df.index.get_level_values('right').map({OptionRight.call: 1, OptionRight.put: 0}).values
    df['delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['mid_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    df['bid_delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['bid_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    df['ask_delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['ask_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    df['vega_mid_iv'] = 100 * get_vega_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['mid_iv'].values, dividends=dividends, calculation_date=df['date'].values)

    ts_delta = df.index.get_level_values('expiry').to_series().apply(
        lambda ex: datetime(ex.year, ex.month, ex.day)).values - df.index.get_level_values('ts').to_series().values
    df['dte'] = (ts_delta / 1000 ** 3).astype(float) / (60 ** 2 * 24 * 365)
    df['dte_int'] = (df['dte'] * 365).apply(math.floor)  # after noon 12:00 dte is > 0.5, would want to cut off day then already

    v_strike = df['strike_flt'].values
    df['moneyness'] = v_strike / df['spot']

    df['moneyness_fwd'] = get_moneyness_fwd(option_frame.equity, v_strike, df['spot'].astype(float).values,
                                            df['tenor'].values,
                                            t0=df.index.get_level_values('ts').to_pydatetime())
    df['moneyness_fwd_ln'] = np.log(df['moneyness_fwd'])

    return option_frame


def enrich_atm_iv(option_frame, net_yield):
    df = option_frame.df_options
    df_equity = option_frame.df_equity
    # dATMIV dS correlation
    ps_atm_iv = df2atm_iv(df, df_equity, net_yield=net_yield)
    df['atm_iv'] = None
    enrich_atm_iv_by_right(df)
    # ATM IV can be measured in few ways, most importantly which expiries to include and how to weight them.
    for ts, sub_df in df.groupby(level='ts'):
        # refactor to calculate any moneyness IV. 90, 100, 110
        df.loc[(ts, slice(None), slice(None), slice(None)), 'atm_iv'] = ps_atm_iv.loc[ts]

    iv_cols = ['mid_iv', 'atm_iv']
    for right, sub_df in df.groupby(level='right'):
        for strike, ss_df in sub_df.groupby(level='strike'):
            for expiry, sss_df in ss_df.groupby(level='expiry'):
                sss_df.sort_index(level='ts', inplace=True)
                sss_df['d_atm_iv'] = sss_df['atm_iv'] - sss_df['atm_iv'].shift(1)
                df.loc[sss_df.index, 'd_atm_iv'] = sss_df['d_atm_iv']
                sss_df['dS'] = sss_df['spot'] - sss_df['spot'].shift(1)
                df.loc[sss_df.index, 'dS'] = sss_df['dS']
                sss_df['dP'] = sss_df['mid_price'] - sss_df['mid_price'].shift(1)
                sss_df['dT'] = ((sss_df.reset_index().set_index('ts', drop=False)['ts'] - sss_df.reset_index().set_index('ts', drop=False)['ts'].shift(
                    1)).apply(lambda x: x.seconds / (3600 * 24)) / 365).values
                for iv in iv_cols:
                    sss_df[f'dIV_{iv}'] = (sss_df[iv] - sss_df[iv].shift(1)).values
                    option_frame.df_options.loc[sss_df.index, f'dIV_{iv}'] = sss_df[f'dIV_{iv}']

                option_frame.df_options.loc[sss_df.index, 'dP'] = sss_df['dP']
                option_frame.df_options.loc[sss_df.index, 'dS'] = sss_df['dS']
                option_frame.df_options.loc[sss_df.index, 'dT'] = sss_df['dT']
    return option_frame


def enrich_trades(underlying: Equity, option_frame):
    df = option_frame.df_option_trades

    df['strike_flt'] = df.index.get_level_values('strike').astype(float)
    df['tenor'] = get_v_tenor_from_index(df)
    df['date'] = df.index.get_level_values('ts').date

    v_calc_date = np.array(list(map(lambda x: x.date(), df.index.get_level_values('ts').to_pydatetime())))
    assert len(set(v_calc_date)) == 1, 'The current dividends function only allows a single calculation date'

    dividends = get_dividends(underlying.symbol.upper(), start=v_calc_date[0])

    df['intrinsic_value'] = df.apply(partial(ps2intrinsic_value, spot_column='spot'), axis=1)
    df['fill_iv'] = df2iv(df, price_col_nm='close', dividends=dividends, calculation_date=df['date'].values)
    df['extrinsic_value'] = df['close'] - df['intrinsic_value']

    # assert positive IV derived for positive extrinsic value. Probably need to adapt for very tiny extrinsic values
    # assert len(df[(df['extrinsic_value'] > df['spot'] * 0.01) & (df['fill_iv'].fillna(0) == 0)]) == 0

    df['fill_delta'] = None
    v_is_call = df.index.get_level_values('right').map({OptionRight.call: 1, OptionRight.put: 0}).values
    df['delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['fill_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    df['vega_fill_iv'] = 100 * get_vega_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['fill_iv'].values, dividends=dividends, calculation_date=df['date'].values, equity=underlying)

    ts_delta = df.index.get_level_values('expiry').to_series().apply(
        lambda ex: datetime(ex.year, ex.month, ex.day)).values - df.index.get_level_values('ts').to_series().values
    df['dte'] = (ts_delta / 1000 ** 3).astype(float) / (60 ** 2 * 24 * 365)
    df['dte_int'] = (df['dte'] * 365).apply(math.floor)  # after noon 12:00 dte is > 0.5, would want to cut off day then already

    v_strike = df.index.get_level_values('strike').astype(float).values
    df['moneyness_fwd'] = get_moneyness_fwd(option_frame.equity, v_strike, df['spot'].astype(float).values,
                                            df['tenor'].values,
                                            t0=df.index.get_level_values('ts').to_pydatetime())
    df['moneyness_fwd_ln'] = np.log(df['moneyness_fwd'])

    # Add tick direction
    # df['spread'] = df['ask_close'] - df['bid_close']
    # df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
    # df['tick_direction'] = None
    # df['tick_direction'] = df['tick_direction'].mask(df['close'] > df['mid_price'], TickDirection.up)
    # df['tick_direction'] = df['tick_direction'].mask(df['close'] < df['mid_price'], TickDirection.down)
    # # Remaining fills equal to mid price. Fill them alternatingly with up and down ticks
    # ix_na = df['tick_direction'].isna()
    # n_rows = len(df.loc[ix_na])
    # df.loc[ix_na, 'tick_direction'] = np.array([TickDirection.up, TickDirection.down] * n_rows)[:n_rows]

    return option_frame


def stressed_iv_col_nm(ds_ret) -> str: return f'iv_ds_ret_{ds_ret:.2f}'
def stressed_npv_col_nm(ds_ret) -> str: return f'npv_ds_ret_{ds_ret:.2f}'


@time_it
def enrich_skew_measures(option_frame, equity: Equity):
    df = option_frame.df_options
    iv_surface = IVSurface(equity, df)

    iv_surface.enrich_iv_curvature(iv_col='mid_iv')
    iv_surface.df = enrich_regressed_skew_rolling(iv_surface.df)
    v_ds_ret = np.linspace(0.8, 1.2, 21)
    iv_surface.mp_enrich_mean_regressed_skew_ds(v_ds_ret)


def fetch_option_frames_by_separate_dt_spans(dates: List[date], underlying: Equity, resolution: Resolution, seq_ret_threshold: float):
    frames = []
    dt_span = []
    for dt in dates:
        if not dt_span:
            dt_span.append(dt)
            continue
        if dt - dt_span[-1] == pd.Timedelta(days=1):
            dt_span.append(dt)
            continue
        else:
            info(f'Fetching: {dt_span}')
            option_frame = generate_option_frame(dt_span[0], dt_span[-1], underlying, resolution, seq_ret_threshold)
            frames.append(option_frame)
        dt_span = [dt]
    info(f'Fetching: {dt_span}')
    option_frame = generate_option_frame(dt_span[0], dt_span[-1], underlying, resolution, seq_ret_threshold)
    frames.append(option_frame)

    df_quotes = pd.concat([frame.df_options for frame in frames]).sort_index()
    df_trades = pd.concat([frame.df_option_trades for frame in frames]).sort_index()
    df_equities = pd.concat([frame.df_equity for frame in frames]).sort_index()

    return OptionFrame(
        equity=underlying,
        df_options=df_quotes,
        df_equity=df_equities,
        df_option_trades=df_trades,
        start=dates[0],
        end=dates[-1],
        resolution=resolution,
        seq_ret_threshold=seq_ret_threshold,
        version='1',
        ts_created=datetime.now()
    )


def get_option_frame(underlying: Equity, dates: List[date], columns_quote: List[str] = (), columns_trade: List[str] = (), resolution: Resolution = Resolution.second,
                     seq_ret_threshold: float = 0.002) -> OptionFrame:
    frames = []
    for dt in dates:
        frames.append(get_option_frame_by_date(underlying, [dt], columns_quote, columns_trade, resolution, seq_ret_threshold))
    df_quotes = pd.concat([frame.df_options for frame in frames]).sort_index()
    df_option_trades = pd.concat([frame.df_option_trades for frame in frames]).sort_index()
    df_equity = pd.concat([frame.df_equity for frame in frames]).sort_index()
    return OptionFrame(
        equity=underlying,
        df_options=df_quotes,
        df_equity=df_equity,
        df_option_trades=df_option_trades,
        start=dates[0],
        end=dates[-1],
        resolution=resolution,
        seq_ret_threshold=seq_ret_threshold,
        version='1',
        ts_created=datetime.now()
    )


@cache_to_disk('option_frame', Paths.path_analysis_frames)
def get_option_frame_by_date(underlying: Equity, dates: List[date], columns_quote: List[str] = (), columns_trade: List[str] = (), resolution: Resolution = Resolution.second,
                     seq_ret_threshold: float = 0.01) -> OptionFrame:
    option_frame = fetch_option_frames_by_separate_dt_spans(dates, underlying, resolution, seq_ret_threshold)

    option_frame = enrich_quotes(underlying, option_frame)
    option_frame = enrich_trades(underlying, option_frame)

    return option_frame


def check_data_presence(sym_date: SymDate, scope: ScopePrePost|str) -> bool:
    client = Client()
    all_present = True
    ticker = sym_date.symbol
    release_date = sym_date.date

    for dt in scoped_dates(release_date, scope):
        for sec_type in (SecurityType.option, SecurityType.equity):
            res = client.date_present(Equity(ticker), dt, sec_type)
            all_present = all_present and res
            if not res:
                print(f'{ticker} {dt} {sec_type} missing')
                return False

    return all_present


def available_sym_dates(tickers: str = None, scope: str|ScopePrePost = ScopePrePost.mini_train) -> List[SymDate]:
    tickers_ = tickers or ','.join(os.listdir(Paths.path_data.joinpath(USA, SECOND))).upper()
    sym_dates = []
    for sym in tickers_.split(','):
        for release_date in EarningsPreSessionDates(sym):
            if check_data_presence(SymDate(sym, release_date), scope) or release_date >= date.today():
                sym_dates.append(SymDate(sym, release_date))
    return sym_dates


def run():
    tickers = ','.join(os.listdir(os.path.join(Paths.path_data, r'option\usa\second'))).upper()
    tickers = 'ORCL'

    for take in [-2]:
        for sym in tickers.split(','):
            try:
                release_date = EarningsPreSessionDates(sym)[take]
                print(release_date)
            except IndexError:
                print(f'{sym} has no earnings date for take={take}.')
                continue

            start, end = earnings_download_dates_start_end(release_date)
            if not check_data_presence(SymDate(sym, release_date)):
                print(f'{sym} data not present.')
                continue
            # if release_date >= date.today():
            #     print(f'{sym} earnings date is in the future.')
            #     continue
            # easy override for earnings announcement dates

            equity = Equity(sym)
            resolution = Resolution.minute
            seq_ret_threshold = 0.002
            # resolution = Resolution.second
            # seq_ret_threshold = 0.001
            v = year_quarter(end)

            rate = DiscountRateMarket
            dividend_yield = get_dividend_yield(sym)

            # if OptionFrame.fn(sym.upper(), resolution, seq_ret_threshold, year_quarter(end)) in os.listdir(Paths.path_analysis_frames):
            #     print(f'{sym} already processed.')
            #     continue

            # option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, v)
            option_frame = generate_option_frame(start, end, equity, resolution, seq_ret_threshold, version=v)
            option_frame = enrich_quotes(equity, option_frame)
            # option_frame = enrich_atm_iv(option_frame, net_yield)
            option_frame = enrich_trades(equity, option_frame)

            # kalman_states = fit_kalman_states(option_frame.df_option_trades, release_date, net_yield)
            # enrich_kalman(option_frame.df_option_trades, kalman_states, net_yield=net_yield)
            # enrich_skew_measures(option_frame, equity)
            # option_frame.store()

            print(f'Done. {sym.upper()} {seq_ret_threshold} {v}')


if __name__ == '__main__':
    # run()
    """
    Worked for a while. Increasingly in need to disparate dates, immediate availability without storing this, wanting to just get the needed columns.
    So a function where one passes: ticker, dates, column names.
    """
