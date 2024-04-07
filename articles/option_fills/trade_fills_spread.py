from functools import partial
from typing import Dict

import pandas as pd
import QuantLib as ql

from itertools import chain
from datetime import date, timedelta, datetime
import plotly.graph_objs as go
from options.client import Client
from options.helper import ps2iv, ps2delta, quotes2multi_index_df
from options.typess.enums import TickType, Resolution, SecurityType
from options.typess.equity import Equity
from options.typess.option_contract import OptionContract
from shared.constants import DiscountRateMarket, DividendYield
from shared.modules.logger import logger
from shared.plotting import plot_ps_trace, show
from markdown import markdown as md


def run():
    """
    Given a trade count and a time period, find the % of spread between 2 borders (bid/ask, ewma(bid/ask), etc.)

    For this, loading trade data only, plot fill as % of spread between 2 borders (bid/ask, ewma(bid/ask), etc.)
    Plot how this metric rolls over time, essentially calibrating an estimator.
    """
    syms = [Equity(s) for s in ['DELL']]
    # load all option trades in between
    period = timedelta(days=7)
    start = date(2023, 11, 1)
    end = date(2023, 12, 31)
    resolution = Resolution.second

    client = Client()
    for equity in syms:
        eq_trades = client.history([equity], start, end, resolution, TickType.trade, SecurityType.equity)
        ps_spot = eq_trades[str(equity)]['close']

        contracts = list(chain(*client.get_contracts(equity, as_of=start).values()))
        trades = client.history(contracts, start, end, resolution=resolution, tick_type=TickType.trade, security_type=SecurityType.option)
        trades = {OptionContract.from_contract_nm(k): v for k, v in trades.items()}

        quotes = client.history(contracts, start, end, resolution, TickType.quote, SecurityType.option)
        quotes: Dict[OptionContract, pd.DataFrame] = {OptionContract.from_contract_nm(k): v for k, v in quotes.items()}

        dct_c_df = {}
        for c, dfq in quotes.items():
            if c not in trades:
                continue
            dft = trades[c]
            df = dfq.merge(dft.rename(columns={'close': 'fill_price'}), how='outer', right_index=True, left_index=True)
            df = df.merge(ps_spot.rename('spot'), how='outer', right_index=True, left_index=True)
            for col in ['spot', 'bid_close', 'ask_close']:
                df[col].ffill(inplace=True)
            ix_keep = df.index[df['fill_price'].notna()]
            df = df.loc[ix_keep]
            df['option_contract'] = c
            df.index.rename('index', inplace=True)
            df.drop(['bid_open', 'bid_high', 'bid_low', 'ask_open', 'ask_high', 'ask_low', 'open', 'high', 'low', 'bid_size', 'ask_size'], axis=1, inplace=True)
            dct_c_df[c] = df

        df = quotes2multi_index_df(dct_c_df)

        # Dropping expired derivatives
        ix_drop = []
        for ts, sub_df in df.groupby(level='ts'):
            for expiry, sub_sub_df in sub_df.groupby(level='expiry'):
                if (expiry < ts.date()):
                    ix_drop.extend(sub_sub_df.index)
        df.drop(ix_drop, inplace=True)

        # Deriving implied volatilities
        rate = DiscountRateMarket
        dividend = DividendYield[str(equity)]
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        day_count = ql.Actual365Fixed()
        df['bid_iv'] = df.apply(partial(ps2iv, price_col='bid_close', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
        df['ask_iv'] = df.apply(partial(ps2iv, price_col='ask_close', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
        df['bid_delta'] = df.apply(partial(ps2delta, iv_col='bid_iv', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
        df['ask_delta'] = df.apply(partial(ps2delta, iv_col='ask_iv', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
        df['mid_delta'] = (df['bid_delta'] + df['ask_delta']) / 2
        df['spread'] = df['ask_close'] - df['bid_close']
        df['fill_as_%_of_spread'] = 100*(df['fill_price'] - df['bid_close']) / df['spread']
        df['direction'] = None
        df['direction'] = df['direction'].mask(df['fill_as_%_of_spread'] < 50, 'sell')
        df['direction'] = df['direction'].mask(df['fill_as_%_of_spread'] > 50, 'buy')
        ix_50_sell = df.index[df['direction'].isna()].values[::2]
        df.loc[ix_50_sell, 'direction'] = 'sell'
        df['direction'].fillna('buy', inplace=True)
        assert df['direction'].isna().sum() == 0

        df['moneyness'] = df['spot'] / df.index.get_level_values('strike').astype(float)
        df['option_contract_str'] = df['option_contract'].values.astype(str)
        df['tenor'] = (df.index.get_level_values('expiry').to_series().reset_index(drop=True) - pd.Series(df.index.get_level_values('ts').to_pydatetime()).apply(
            lambda x: x.date())).apply(lambda x: x.days / 365).values

    # now start analyzing
    r = []
    # rm outlier
    len0 = len(df)
    df = df.loc[df['fill_as_%_of_spread'] <= 110]
    df = df.loc[df['fill_as_%_of_spread'] > -10]
    logger.info(f'rm {100 - 100*len(df) / len0}%; #{len0 - len(df)} outliers')

    df.hist(column='fill_as_%_of_spread', by='direction', bins=50)
    # Majority is at 0, 50 and 100%, each fairly equal parts
    # Now work out how this behaves as f(moneyness)

    fig = plot_ps_trace(go.Scatter(x=df['moneyness'], y=df['volume'], mode='markers', marker=dict(size=4), name='volume'))
    fig.update_layout(title=f'Volume by moneyness for {equity} options', xaxis_title='Moneyness S/K', yaxis_title='Volume')
    show(fig, fn='vol_by_moneyness.html')
    r += ['vol_by_moneyness.html']
    r += [md('As expected, volume is highest at ATM, and decreases as we move away from ATM.')]

    fig = plot_ps_trace(
        go.Scatter(x=abs(df['mid_delta']), y=df['volume'], mode='markers', marker=dict(size=4), name='volume')
    )
    fig.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='Delta', yaxis_title='Volume')
    show(fig, fn='vol_by_delta.html')
    r += ['vol_by_delta.html']

    # traces = [go.Scatter(x=df['moneyness'], y=df['fill_as_%_of_spread'], mode='markers', marker=dict(size=4), name='fill_as_%_of_spread')]

    # Given a cutoff volume x per Period and fill_%, what is the min/max moneyness or delta worth pursuing. easier to use delta, encompasses expiry and moneyess
    # assumption. cut 50% in half for buy sell. 0/100
    ts = df.index.get_level_values('ts')

    df = df.sort_index(level='ts')
    for col in ['volume_rolled_sum']:
        df[col] = None
    for o, s_df in df.groupby('option_contract_str'):
        s_df['volume_rolled_sum'] = s_df[['volume']].reset_index().sort_values('ts').rolling(window=period, on='ts').sum('volume')['volume'].values
        df.loc[s_df.index, 'volume_rolled_sum'] = s_df['volume_rolled_sum']

    # for o, s_df in df.groupby('option_contract_str'):
    #     # print(o)
    #     fig = plot_ps_trace(go.Scatter(x=s_df.index.get_level_values('ts'), y=s_df['volume_rolled_sum'], mode='markers', marker=dict(size=4), name=o))
    #     fig.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='TS', yaxis_title='Volume')
    #     show(fig, fn=f'fig{o}.html')

    dft = df.loc[df.index.get_level_values('ts').min() + period:]

    fig = plot_ps_trace(
        go.Scatter(x=dft['tenor'], y=dft['volume_rolled_sum'], mode='markers', marker=dict(size=4), name='volume_rolled_sum')
    )
    fig.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='Tenor', yaxis_title='Volume')
    show(fig, fn='tenor.html')

    # Rolled volume by delta, tenor  --- down the road, cnt_fills_over_period = f(delta, tenor) a regressor. Further, what's the fill_as_%_of_spread for these, can define as sth gaussian? mean, std, skew, kurtosis.
    # So expected volume, a mean % fill, an std, should be able to calc some prob(fill | % spread)...
    # So I need both, decent volume and decent fill_as_%_of_spread. Latter may not be an issue and same distibution no matter volume. check the mean.. also a surface by tenor, delta then...
    # That'll determine target price, IV, etc...

    dft['cut'] = pd.cut(dft['tenor'], bins=20)
    for cut, s_df in dft.groupby('cut'):
        fig = plot_ps_trace(
            go.Scatter(x=abs(s_df['mid_delta']), y=s_df['volume_rolled_sum'], mode='markers', marker=dict(size=4), name='volume_rolled_sum')
        )
        fig.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='Delta', yaxis_title='Volume')
        show(fig, fn=f'vol_by_delta{cut}.html')

    df[df['volume_rolled_sum'] <= 1]['volume_rolled_sum']

    # groupby week

    df['week'] = df.index.get_level_values('ts').to_series().apply(lambda x: x.week).values
    df.groupby('week')['volume_rolled_sum'].mean().plot()
    for week, s_df in df.groupby('week'):
        
        fig = plot_ps_trace(
            go.Scatter(x=s_df['mid_delta'], y=s_df['volume_rolled_sum'], mode='markers', marker=dict(size=4), name='volume_rolled_sum')
        )
        fig.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='Tenor', yaxis_title='Volume')
        show(fig, fn=f'week{week}.html')


if __name__ == "__main__":
    run()
