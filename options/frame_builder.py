import math
import os.path
import pandas as pd
import numpy as np

from importlib import reload
from functools import partial, reduce
from typing import List
from itertools import chain
from datetime import date, time, datetime

from common.modules import resolution
from options.types.scope_pre_post import ScopePrePost, scoped_dates # JL
from options.types.dividend import get_dividends # JL
from options.types.option import get_vega_cuda, get_delta_cuda # JL
from options.types.sym_date import SymDate # JL
from shared.modules.logger import info, error
from shared.paths import Paths # JL
from shared.constants import EarningsPreSessionDates, USA, SECOND # JL
from options.helper import (find_loc_every_x_pc,
                            spot_from_df_equity_into_options, \
                            ps2intrinsic_value,
                            quotes2multi_index_df,
                            load_option_trades,
                            join_spot,
                            join_quotes,
                            ps2mid_iv_if_nonzero, \
                            get_moneyness_fwd, # JL
                            cache_to_disk,
                            get_v_tenor_from_index,
                            df2iv, #JL
                            get_pkl_cache_key # not a good naming system anyway
                            )
from options.types.enums import SecurityType, TickType, Resolution, OptionRight #JL
from options.types.equity import Equity #JL
from options.client import Client #JL as QCPrices
from options.types.option_contract import OptionContract  #JL
from options.types.option_frame import OptionFrame  #JL
import options.client as mClient #JL as QCPrices

reload(mClient)
client = mClient.Client()

def generate_option_frame(start, end, equity, resolution, seq_ret_threshold, version='1', keep_seconds: List[datetime] = None):
    client = Client()
    trades_eq = client.history([equity], start, end, resolution, TickType.trade, SecurityType.equity)
    if str(equity) not in trades_eq:
        error(f'No equity trades for {equity} in {start} - {end}.')

    ps_spot = trades_eq[str(equity)]['close']
    # removing non-RTH hours from spot because not present in quotes
    ps_spot = ps_spot[(ps_spot.index.time >= time(9, 30)) & (ps_spot.index.time <= time(16, 0))]
    print(f"equity_history: len: {(len(ps_spot))} sum close {sum(ps_spot)}")

    # refactor to simply loading all of them...
    # contracts = client.central_volatility_contracts(equity, start, end, n=3000)
    # contracts = list(chain(*contracts.values()))

    contracts = client.get_all_contracts(equity, start, end)
    print(f'# Contracts loaded: {len(contracts)}') # 3373

    quotes = client.history(contracts, start, end, resolution, TickType.quote, SecurityType.option)
    quotes = {OptionContract.from_contract_nm(k): v for k, v in quotes.items()}
    n_rows = reduce(lambda x, y: x + len(y), quotes.values(), 0)
    print(f"option_quotes_raw: # rows: {n_rows}")
    if not quotes:
        error(f'No option quotes for {equity} in {start} - {end}.')

    trades = load_option_trades(contracts, start, end, resolution)
    n_rows = reduce(lambda x, y: x + len(y), trades.values(), 0)
    print(f"option_trades_raw: # rows: {n_rows}")

    if not trades:
        error(f'No option trades for {equity} in {start} - {end}.')
        df_trades = None
    else:
        trades = join_spot(trades, ps_spot)  # 36531 -> 36531
        trades = join_quotes(trades, quotes)  # quotes: 9112156  -> trades: 36531 -> 36340

        print(f'''After joining ps_spot: 
              quotes: {reduce(lambda x, y: x + len(y), quotes.values(), 0)}
              trades: {reduce(lambda x, y: x + len(y), trades.values(), 0)}
              ''')
        df_trades = quotes2multi_index_df(trades) # -> 36340

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
    print(f'''After insert ps_spot timestamps: 
      quotes: {reduce(lambda x, y: x + len(y), quotes.values(), 0)}
    ''')

    ps_spot = ps_spot.loc[loc]
    print(f"equity_history: len: {(len(ps_spot))} sum close {(sum(ps_spot))}")
    df = quotes2multi_index_df(quotes)

    # Dropping expired derivatives - check the speed of this... check by ts.date() not every single ts...
    ix_drop = []
    for ts, sub_df in df.groupby(level='ts'):
        for expiry, sub_sub_df in sub_df.groupby(level='expiry'):
            if expiry < ts.date():
                ix_drop.extend(sub_sub_df.index)
    df.drop(ix_drop, inplace=True)
    print(f'''After dropping expired options: 
          quotes: {reduce(lambda x, y: x + len(y), quotes.values(), 0)}
        ''')

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

    print('frame builder.enrich_quotes(): bid iv...')
    df['bid_iv'] = df2iv(df, price_col_nm='bid_close', dividends=dividends, calculation_date=df['date'].values)
    print('frame builder.enrich_quotes(): ask iv...')
    df['ask_iv'] = df2iv(df, price_col_nm='ask_close', dividends=dividends, calculation_date=df['date'].values)
    df['mid_iv'] = df.apply(ps2mid_iv_if_nonzero, axis=1)
    df['extrinsic_bid'] = df['bid_close'] - df['intrinsic_value']
    df['extrinsic_ask'] = df['ask_close'] - df['intrinsic_value']

    # df['vega_mid_iv'] = 100 * get_vega(df['spot'].values, df['strike_flt'].values, df['tenor'].values, df['mid_iv'].values, rate, div_amounts, div_times)
    # df['vega_mid_iv'] = df.apply(partial(ps2vega, iv_col='mid_iv', calendar=calendar, day_count=day_count, dividend=dividend), axis=1)
    # (abs(df['vega_mid_iv'] / 100 - v_vega) < 0.0001).sum() == len(df)

    df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
    print('frame builder.enrich_quotes(): mid_price_iv...')
    df['mid_price_iv'] = df2iv(df, price_col_nm='mid_price', dividends=dividends, calculation_date=df['date'].values, equity=underlying)

    df['bid_delta'] = None
    df['ask_delta'] = None
    df['delta'] = None

    v_is_call = df.index.get_level_values('right').map({OptionRight.call: 1, OptionRight.put: 0}).values
    print('frame builder.enrich_quotes(): delta...')
    df['delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['mid_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    print('frame builder.enrich_quotes(): bid_delta...')
    df['bid_delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['bid_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    print('frame builder.enrich_quotes(): ask_delta...')
    df['ask_delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['ask_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    print('frame builder.enrich_quotes(): vega_mid_iv...')
    df['vega_mid_iv'] = 100 * get_vega_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['mid_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    print('frame builder.enrich_quotes(): vega_mid_iv. done.')

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


def enrich_trades(underlying: Equity, option_frame):
    df = option_frame.df_option_trades

    df['strike_flt'] = df.index.get_level_values('strike').astype(float)
    df['tenor'] = get_v_tenor_from_index(df)
    df['date'] = df.index.get_level_values('ts').date

    v_calc_date = np.array(list(map(lambda x: x.date(), df.index.get_level_values('ts').to_pydatetime())))
    assert len(set(v_calc_date)) == 1, 'The current dividends function only allows a single calculation date'

    dividends = get_dividends(underlying.symbol.upper(), start=v_calc_date[0])

    df['intrinsic_value'] = df.apply(partial(ps2intrinsic_value, spot_column='spot'), axis=1)
    print('frame builder.enrich_trades(): fill_iv...')
    df['fill_iv'] = df2iv(df, price_col_nm='close', dividends=dividends, calculation_date=df['date'].values)
    df['extrinsic_value'] = df['close'] - df['intrinsic_value']

    # assert positive IV derived for positive extrinsic value. Probably need to adapt for very tiny extrinsic values
    # assert len(df[(df['extrinsic_value'] > df['spot'] * 0.01) & (df['fill_iv'].fillna(0) == 0)]) == 0

    df['fill_delta'] = None
    v_is_call = df.index.get_level_values('right').map({OptionRight.call: 1, OptionRight.put: 0}).values
    print('frame builder.enrich_trades(): delta...')
    df['delta'] = get_delta_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['fill_iv'].values, dividends=dividends, calculation_date=df['date'].values)
    print('frame builder.enrich_trades(): vega_fill_iv...')
    df['vega_fill_iv'] = 100 * get_vega_cuda(s=df['spot'].values, k=df['strike_flt'].values, t=df['tenor'].values, v_is_call=v_is_call, iv=df['fill_iv'].values, dividends=dividends, calculation_date=df['date'].values, equity=underlying)
    print('frame builder.enrich_trades(): vega_fill_iv. Done')

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

def downsample_df_options(df, n: int = 3_000) -> 'pd.DataFrame':
    """
    Multi-index of ['ts', 'expiry', 'strike', 'right'].
    Limits each expiry group to at most `n` rows via random sampling.
    """
    info(f'downsample_df_options(): downsampling to {n} rows per tenor...')
    def _sample(group):
        if len(group) <= n:
            return group
        return group.sample(n=n, random_state=None)

    return df.groupby('expiry', group_keys=False).apply(_sample)

def fetch_option_frames_by_separate_dt_spans(dates: List[date], underlying: Equity, resolution: Resolution, seq_ret_threshold: float, n_samples_per_tenor = 3_000):
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
    option_frame.df_options = downsample_df_options(option_frame.df_options, n_samples_per_tenor)
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
                     seq_ret_threshold: float = 0.002, n_samples_per_tenor=3_000) -> OptionFrame:
    frames = []
    for dt in dates:
        fn = get_pkl_cache_key('option_frame', underlying, [dt], columns_quote, columns_trade, resolution, seq_ret_threshold, n_samples_per_tenor)
        fp = os.path.join(Paths.path_analysis_frames, fn)
        # if os.path.exists(fp):
        #     os.remove(fp)
        #     print(f'Deleted {fp}')

        frames.append(get_option_frame_by_date(underlying, [dt], columns_quote, columns_trade, resolution, seq_ret_threshold, n_samples_per_tenor))
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
                             seq_ret_threshold: float = 0.01,
                             n_samples_per_tenor=3_000) -> OptionFrame:
    option_frame = fetch_option_frames_by_separate_dt_spans(dates, underlying, resolution, seq_ret_threshold, n_samples_per_tenor)

    option_frame = enrich_quotes(underlying, option_frame)
    option_frame = enrich_trades(underlying, option_frame)

    return option_frame


def check_data_presence(sym_date: SymDate, scope: ScopePrePost|str) -> bool:
    client = Client()
    all_present = True
    ticker = sym_date.symbol
    release_date = sym_date.date

    for dt in scoped_dates(release_date, scope):
        if dt >= date.today():
            continue
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

if __name__ == '__main__':
    from datetime import date
    from options.types.enums import SecurityType, TickType, Resolution
    from options.types.equity import Equity  # JL
    from options.client import Client  # JL as QCPrices
    equity = Equity("ORCL")
    start = date(2026, 3, 9)
    end = date(2026, 3, 9)
    seq_ret_threshold = 0.002
    resolution = Resolution.second
    # trades_eq = Client().history([equity], start, end, Resolution.second, TickType.trade, SecurityType.equity)
    # trades_eq2 = Client().history([equity], start, end, Resolution.minute, TickType.trade, SecurityType.equity)
    # len(trades_eq[equity.symbol])
    # len(trades_eq2[equity.symbol])
    get_option_frame(equity, [start], resolution=Resolution.second, seq_ret_threshold= 0.002)