import numpy as np
import QuantLib as ql
import pandas as pd
from sklearn import metrics
from functools import partial
from decimal import Decimal
from datetime import timedelta, date, time

from options.helper import repair_prices, ps2mid_iv_if_nonzero, str2ql_option_right, to_ql_dt, \
    get_tenor, ps2npv, get_dividend_yield, df2iv
from options.types.dividend import get_dividends
from options.types.enums import OptionRight
from options.types.equity import Equity
from shared.constants import DiscountRateMarket
from shared.modules.logger import info
from shared.plotting import show, plot_scatter_3d
import plotly.graph_objs as go


def ps2decimal(ps: pd.Series) -> pd.Series:
    return ps.astype(float).round(2).apply(lambda x: Decimal(f'{x:.2f}'))


def ts_closest_to(timestamps, hour=12):
    return timestamps[np.argmin(np.abs([t.time().hour - hour for t in timestamps]))]


def get_calibration_set(df: pd.DataFrame, right: OptionRight, iv_col_nm='mid_iv') -> ql.CalibrationSet:
    calibrationSet = ql.CalibrationSet()

    for expiry, s_df in df.groupby('expiry'):
        exercise = ql.EuropeanExercise(to_ql_dt(expiry))

        for strike, ss_df in s_df.groupby('strike'):
            iv = ss_df[iv_col_nm].mean()
            if np.isnan(iv):
                continue
            payoff = ql.PlainVanillaPayoff(str2ql_option_right(right), float(strike))
            #  calibration_params.calendar.advance(calculation_date, ql.Period(dte(calculation_ts.date(), expiry), ql.Days))
            calibrationSet.push_back((ql.VanillaOption(payoff, exercise), ql.SimpleQuote(iv)))

    return calibrationSet


def get_ah_vol_surface(df: pd.DataFrame, right, calc_date: date, rate: float, dividend_yield: float) -> ql.AndreasenHugeVolatilityAdapter:
    if 'ts' in df.index.names:
        raise ValueError('Index must not contain ts. Provide a snapshot.')
    if 'right' in df.index.names:
        raise ValueError('Index must not contain right. Provide one side.')

    spot = df['spot'].values[0]
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
    calibration_set = get_calibration_set(df, right)

    calculation_date_ql = to_ql_dt(calc_date)
    day_count = ql.Actual365Fixed()
    yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date_ql, rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date_ql, dividend_yield, day_count))

    ah_vol_interpolation = ql.AndreasenHugeVolatilityInterpl(calibration_set, spot_quote, yield_ts, dividend_ts)
    ah_vol_surface = ql.AndreasenHugeVolatilityAdapter(ah_vol_interpolation)
    return ah_vol_surface


def ah2iv(ah_vol_surface, expiry, strike):
    try:
        return ah_vol_surface.blackVol(to_ql_dt(expiry), strike, True)
    except RuntimeError:
        return np.nan


def drop_outliers_from_frame_by_ah_srf(ss_df: pd.DataFrame, right: OptionRight, calc_date: date, rate: float, dividend_yield: float, n_std=3):
    ah_vol_surface = get_ah_vol_surface(ss_df, right, calc_date, rate, dividend_yield)
    expiries = ss_df.index.get_level_values('expiry')
    strikes = ss_df.index.get_level_values('strike').astype(float)
    ivs = ss_df['mid_iv'].values

    ivs_ah = [ah2iv(ah_vol_surface, expiry, strike) for expiry, strike in zip(expiries, strikes)]
    plot_scatter_3d(strikes, expiries, ivs, fn=f'ivs_{right}.html')
    plot_scatter_3d(strikes, expiries, ivs_ah, fn=f'ivs_ah_{right}.html')
    residuals = ivs - ivs_ah
    an_residuals = residuals[~np.isnan(residuals)]
    z_residuals = (residuals - an_residuals.mean()) / an_residuals.std()
    ix_outliers = np.abs(z_residuals) > n_std | np.isnan(residuals)
    return ss_df.drop(ss_df.index[ix_outliers])


def drop_outliers(equity: Equity, df: pd.DataFrame, n_std=5):
    rate = DiscountRateMarket
    dividend_yield = get_dividend_yield(equity)

    new_dfs = []
    for ts, s_df in df.groupby('ts'):
        calc_date = ts.date()

        for right, ss_df in s_df.droplevel('ts').groupby('right'):
            ss_df = ss_df.droplevel('right')
            new_size_df = 0
            size_df = len(ss_df)
            # while new_size_df != size_df:
            size_df = len(ss_df)
            new_ss_df = drop_outliers_from_frame_by_ah_srf(ss_df, right, calc_date, rate, dividend_yield, n_std)
            new_size_df = len(new_ss_df)
            print(f'Outliers removed {ts} {right}: {size_df - new_size_df}')
            ss_df = new_ss_df

            ss_df['ts'] = ts
            ss_df['right'] = right
            new_dfs.append(ss_df)

    return pd.concat(new_dfs).reset_index().set_index(['ts', 'expiry', 'strike', 'right']).sort_index()


def ivs_from_maxbid_minask_iv(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Naturally removes outliers as large bids / small asks get filled.
    Yields the mean of the most buyer-friendly and seller-friendly IV srf.
    """
    info(f'IVs from max bid and min ask across {len(df_in.index.get_level_values("ts").unique())} snaps.')
    rows = []
    for right, s_df in df_in.groupby('right'):
        for expiry, ss_df in s_df.groupby('expiry'):
            for strike, sss_df in ss_df.groupby('strike'):
                bid_row = sss_df.iloc[sss_df['bid_iv'].argmax()]
                ask_row = sss_df.iloc[sss_df['ask_iv'].argmin()]
                spot = (bid_row['spot'] + ask_row['spot']) / 2
                ts = pd.Timestamp.fromisoformat(bid_row.name[0].date().isoformat())
                mid_price_iv = (bid_row['mid_price_iv'] + ask_row['mid_price_iv']) / 2

                rows.append([ts, expiry, strike, right, bid_row['bid_iv'], ask_row['ask_iv'], spot, mid_price_iv])
    return pd.DataFrame(rows, columns=['ts', 'expiry', 'strike', 'right', 'bid_iv', 'ask_iv', 'spot', 'mid_price_iv']).set_index(['ts', 'expiry', 'strike', 'right'])


def get_moneyness_fwd_ln(df, net_yield) -> pd.Series:
    raise NotImplementedError('current version incorrectly assumes constant yield and no dividends')
    strikes = df.index.get_level_values('strike').astype(float)
    strike_fwd = strikes * np.exp(net_yield * df['tenor'])
    moneyness_fwd = (strike_fwd / df['spot']).astype(float)
    return np.log(moneyness_fwd)


def ivs_from_maxbid_minask_iv_enriched(df_in: pd.DataFrame, equity: Equity) -> pd.DataFrame:
    df = ivs_from_maxbid_minask_iv(df_in)
    df['moneyness'] = df.index.get_level_values('strike').astype(float) / df['spot']
    df['tenor'] = df.apply(lambda ps: get_tenor(ps.name[1], ps.name[0].date()), axis=1)

    rate = DiscountRateMarket
    dividend_yield = get_dividend_yield(equity)
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()
    dividends = get_dividends(equity.symbol.upper(), start=df['date'].values[0])

    df['moneyness_fwd_ln'] = get_moneyness_fwd_ln(df, rate - dividend_yield)
    df['mid_iv'] = (df['bid_iv'] + df['ask_iv']) / 2
    df['bid_close'] = df.apply(partial(ps2npv, iv_col='bid_iv', calendar=calendar, day_count=day_count, dividend=dividend_yield), axis=1)
    df['ask_close'] = df.apply(partial(ps2npv, iv_col='ask_iv', calendar=calendar, day_count=day_count, dividend=dividend_yield), axis=1)
    df['mid_price'] = df['mid_close'] = (df['bid_close'] + df['ask_close']) / 2
    df['mid_price_iv'] = df2iv(df, price_col_nm='mid_price', dividends=dividends, calculation_date=df['date'].values)
    # df['mid_price_iv'] = df.apply(partial(ps2iv, price_col='mid_price', calendar=calendar, day_count=day_count, dividend=dividend_yield), axis=1)
    return df


def repair_bid_ask(df: pd.DataFrame, equity: Equity, plot=False) -> pd.DataFrame:
    dividend_yield = get_dividend_yield(equity)
    net_yield = rate - dividend_yield

    dfs = []
    for ts, s_df in df.groupby(level='ts'):
        calculation_date = s_df.index.get_level_values('ts').date[0]
        dividends = get_dividends(equity.symbol.upper(), start=calculation_date)

        for right, ss_df in s_df.groupby(level='right'):
            if ss_df['ask_close'].notna().sum() < 10:
                print(f'Not enough data for {ts} {right}.')
                continue

            dft = repair_prices(ss_df.reset_index(drop=False), calculation_date, n_repairs=1, plot=plot, right=right,
                                col_nm_map={'mid_close_underlying': 'spot', 'mid_close': 'mid_price'}, net_yield=net_yield)
            dft['ts'] = ts
            dft['right'] = right
            dft['strike'] = dft['strike'].apply(lambda x: Decimal(round(x, 1)))

            # # Check
            # ix = dft.set_index(['ts', 'expiry', 'strike', 'right']).index
            # max(abs(ss_df.loc[ix, 'ask_close'].values / dft['ask_close'].values -1))
            # max(abs(ss_df.loc[ix, 'bid_close'].values / dft['bid_close'].values -1))
            # max(abs(ss_df.loc[ix, 'mid_price'].values / dft['mid_price'].values -1))


            dft['mid_price'] = (dft['ask_close'] + dft['bid_close']) / 2

            dft['bid_iv'] = df2iv(dft, price_col_nm='bid_close', dividends=dividends, calculation_date=calculation_date)
            dft['ask_iv'] = df2iv(dft, price_col_nm='ask_close', dividends=dividends, calculation_date=calculation_date)
            # dft['bid_iv'] = dft.apply(partial(ps2iv, price_col='bid_close', calendar=calendar, day_count=day_count, dividend=dividends), axis=1)
            # dft['ask_iv'] = dft.apply(partial(ps2iv, price_col='ask_close', calendar=calendar, day_count=day_count, dividend=dividends), axis=1)

            dfs.append(dft)

    df_b = pd.concat(dfs).set_index(['ts', 'expiry', 'strike', 'right'])
    df_b['mid_iv'] = df_b.apply(ps2mid_iv_if_nonzero, axis=1)

    return df_b.sort_index()


def snap_diffusion_ivs(df_options: pd.DataFrame, release_date: date, equity: Equity) -> pd.DataFrame:
    ref_dt = release_date - timedelta(days=28)
    df = df_options[df_options.index.get_level_values('ts').date == ref_dt]
    v_ts = df.index.get_level_values('ts').time
    df = df[(v_ts >= time(9, 45)) & (v_ts <= time(16, 45))]
    df = ivs_from_maxbid_minask_iv_enriched(df, equity)

    # ts = ts_closest_to(df.index.get_level_values('ts').unique())
    # df = repair_bid_ask(df, equity)
    # df['moneyness'] = df.index.get_level_values('strike').astype(float) / df['spot']  # Should someday reversed to S/K
    # plot_scatter_3d(df['moneyness'], df['tenor'], df['mid_iv'], fn=f'{equity.symbol}_repaired_maxbid_minask_diffusion_iv.html')
    # df = repair_bid_ask(df.loc[[ts]], equity)

    # Rather than taking the median, may rather want to fit a model like biquadratic spline or heston and check how the parameters change over time.
    # Then can generate surfaces with confidence intervals given the parameters. Flexible!
    return df


def plot_surface(xy, col, axes=('moneyness', 'tenor'), fn=None, scatter=True):
    fig = go.Figure()

    if scatter:
        x_val = xy[axes[0]].values if axes[0] in xy.columns else xy.index.get_level_values(axes[0]).values
        y_val = xy[axes[1]].values if axes[1] in xy.columns else xy.index.get_level_values(axes[1]).values
        fig.add_trace(
            go.Scatter3d(x=x_val, y=y_val, z=xy[col].values, mode='markers', marker=dict(size=2))
        )
    else:
        df = xy.groupby(list(axes)).agg({col: 'mean'}).reset_index().pivot(index=axes[0], columns=axes[1], values=col)
        fig.add_trace(
            go.Surface(x=df.columns.values, y=df.index.values, z=df)
        )

    fig.update_layout(title=fn or f'{col} surface', autosize=True)
    show(fig, fn=fn or f'earnings_iv_drop_surface_{col}_{axes}.html')


def ps_in_slices(ps, slices) -> bool:
    return any((ps['symbol'] in slice and ps['release_date'] in slice for slice in slices))

def score(y, y_pred, weights) -> str:
    md = f'''
        # Samples: {len(y)}  
        R2            : {metrics.r2_score(y, y_pred)}  
        R2  (weighted): {metrics.r2_score(y, y_pred, sample_weight=weights)}  
        MAE           : {metrics.mean_absolute_error(y, y_pred)}  
        MAE (weighted): {metrics.mean_absolute_error(y, y_pred, sample_weight=weights)}  
        RMSE          : {metrics.mean_squared_error(y, y_pred)}  
        RMSE(weighted): {metrics.mean_squared_error(y, y_pred, sample_weight=weights)}  
    '''
    print(md)
    return md