import numpy as np
import pandas as pd

from collections import defaultdict
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta, datetime
from options.surfaces.processors import get_v_ivs, v_ivs_to_weighted_metrics
from options.client import Client
from options.helper import add_trade_days, date_to_eod_expiry, get_v_tenor, date_to_datetime, get_pkl_cache_key
from options.types.SSVIParams import SSVIMetric
from options.types.dividend import get_dividends
from options.types.enums import Resolution, TickType, SecurityType
from options.types.equity import Equity
from options.types.iv_surface_essvi import IVSurface
from options.types.option import get_vega_cuda
from options.types.quote_side import QuoteSide
from options.types.scope_pre_post import scoped_dates, ScopePrePost
from options.types.sym_date import SymDate
from shared.constants import EarningsPreSessionDates
from shared.modules.logger import warning, info
from shared.paths import Paths
from shared.yield_curve import YieldCurve


def pivot_on_start_date(df_in: pd.DataFrame, key=None, max_lag=None) -> pd.DataFrame:
    key = key or ['underlying', 'tenor_dt']
    dfs = []
    for _, s_df in df_in.groupby(['underlying', 'release_date']):
        df = None
        for days_to_release, ss_df in s_df.groupby('business_days_to_release'):
            # Average of all metrics within the same day
            df_mean = ss_df.groupby(['underlying', 'tenor_dt', 'release_date', 'business_days_to_release']).mean().reset_index()
            if df is None:
                df = df_mean.copy()
                continue
            if max_lag is not None and days_to_release >= max_lag:
                break

            df = df.merge(df_mean, on=key, suffixes=('', f'_lag_{days_to_release}'))
        # for i, dt in enumerate(sorted(s_df['ts_end'].unique(), reverse=True)):
        #     df_slice = s_df[s_df['ts_end'] == dt]
        #     if i == 0:
        #         df = df_slice.copy()
        #         continue
        #     if max_lag is not None and i >= max_lag:
        #         break
        #     df = df.merge(df_slice, on=key, suffixes=('', f'_lag_{i}'))
        df = df.rename({f'value_{metric}': f'value_{metric}_lag_0' for metric in ['theta', 'rho', 'psi']}, axis=1)
        dfs.append(df)

    return pd.concat(dfs)


@dataclass
class FeatureConfig:
    transformers: List[BaseTransformer] = field(default_factory=list)

class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def get_feature_columns(self) -> List[str]:
        return []

    def fill_required_cols(self, df: pd.DataFrame, type=float) -> pd.DataFrame:
        for c in set(self.get_feature_columns()).difference(df.columns):
            df[c] = None
            df[c] = df[c].astype(type)
        return df


class EnrichPreviousJumpReleaseJumpTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

class EnrichReleaseDatesTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df['release_date'] = None
        df['release_datetime'] = None
        for sym, gp in df.groupby('underlying'):
            for release_date in sorted(EarningsPreSessionDates(sym)):
                ix_match = gp.index[(gp['ts_end'].dt.date <= release_date + timedelta(days=10)) & (gp['ts_end'].dt.date >= release_date - timedelta(days=50))]
                df.loc[ix_match, 'release_date'] = release_date
                df.loc[ix_match, 'release_datetime'] = datetime(release_date.year, release_date.month, release_date.day, 16)
        return df

    def get_feature_columns(self) -> List[str]:
        return []


class EnrichSpotTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        client = Client()
        v_df = []
        for sym, gp in df.groupby('underlying'):
            dates = sorted(list({ts.date() for ts in gp['ts_end'].unique()}))
            equity = Equity(sym)
            dfs = []
            for dt in dates:
                trades_eq = client.history([Equity(sym)], dt, dt, Resolution.second, TickType.trade, SecurityType.equity)
                df_equity = trades_eq[str(equity)][['close']].rename(columns={'close': 'spot'}).reset_index()
                dfs.append(df_equity)
            df_equity = pd.concat(dfs)

            # df_equity = get_option_frame(Equity(sym), dates, Resolution.second, seq_ret_threshold=0.002).df_equity.rename(columns={'close': 'spot'}).reset_index()
            dft = gp.merge(df_equity.reset_index(), left_on='ts_end', right_on='time', how='outer', validate='many_to_one')
            dft['time'] = dft['time'].fillna(dft['ts_end'])
            dft = dft.sort_values('time')
            dft['spot'] = dft['spot'].ffill().astype(float)
            dft = dft[~dft['ts_end'].isna()]
            assert gp.shape[0] == dft.shape[0]
            v_df.append(dft)
        return pd.concat(v_df)


class EnrichReturnTransformer(BaseTransformer):
    def __init__(self, ret: float = None):
        self.ret = ret

    def transform(self, df: pd.DataFrame, required=('release_date', 'spot'), **kwargs) -> pd.DataFrame:
        if self.ret is not None:
            if abs(self.ret) > 0.5:
                warning(f'EnrichReturnTransformer.transform(): Outsized return. Likely wrong argument: {self.ret}')
            df['return'] = self.ret # For scenario predictions
            return df

        df['post_release_ts'] = df['release_date'].apply(lambda x: date_to_datetime(add_trade_days(x, 1), timedelta(hours=12)))
        client = Client()
        v_df = []
        for sym, gp in df.groupby('underlying'):
            dates = sorted(list({ts.date() for ts in gp['post_release_ts'].unique()}))
            equity = Equity(sym)
            dfs = []
            for dt in dates:
                trades_eq = client.history([Equity(sym)], dt, dt, Resolution.second, TickType.trade, SecurityType.equity)
                df_equity = trades_eq[str(equity)][['close']].rename(columns={'close': 'spot_post'}).reset_index()
                dfs.append(df_equity)
            df_equity = pd.concat(dfs)

            # df_equity = get_option_frame(Equity(sym), dates, Resolution.second, seq_ret_threshold=0.002).df_equity.rename(columns={'close': 'spot'}).reset_index()
            dft = gp.merge(df_equity.reset_index(), left_on='post_release_ts', right_on='time', how='outer', validate='many_to_one')
            dft = dft.sort_values('post_release_ts')
            dft['spot_post'] = dft['spot_post'].ffill().astype(float)
            dft = dft[~dft['post_release_ts'].isna()]
            assert gp.shape[0] == dft.shape[0]
            v_df.append(dft)

        df2 = pd.concat(v_df)
        df2['return'] = df2['spot_post'] / df2['spot'] - 1
        assert len(df) == len(df2)
        return df2

    def get_feature_columns(self) -> List[str]:
        return ['return']


class AddBusinessDaysTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df['time_to_release_days'] = (df['release_datetime'].apply(lambda x: x.timestamp()) - df['ts_end'].apply(lambda x: x.timestamp())) / (3600 * 24)
        return df

    def get_feature_columns(self) -> List[str]:
        return ['time_to_release_days']


class PivotMetricsTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame, key=None, values=None, **kwargs) -> pd.DataFrame:
        key = key or ['underlying', 'release_date', 'tenor_dt', 'metric']
        values = values or [c for c in df.columns if 'theta' in c or 'psi' in c or 'rho' in c]
        key = list(set(df.columns).difference(set(values)))
        df_pivot = df.pivot(index=key, columns=values)

        # df_pivot = df[df['metric'] == 'theta'].copy()
        # df_pivot['value_theta'] = df_pivot['value']
        #
        # for c in ['rho', 'psi']:
        #     df_pivot[f'value_{c}'] = None
        #     df_pivot = df_pivot.merge(df[df['metric'] == c], on=key, suffixes=('', f'_{c}'), validate='one_to_one')
        #

        # assert df_pivot[key].duplicated().sum() == 0
        return df_pivot


class AddIvJumpsTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        v_ivs = kwargs['v_ivs']
        sym_dates = kwargs['sym_dates']
        out = df.copy()
        jump_rows = []
        for ivs in v_ivs:
            ivs_sym_dates = [
                SymDate(ivs.underlying.symbol, add_trade_days(ivs.last_calibration_ts().date(), 1)),
                SymDate(ivs.underlying.symbol, ivs.last_calibration_ts().date()),
            ]
            if any((dt == sd for sd in sym_dates for dt in ivs_sym_dates)):
                jump_rows.append({'surface_id': ivs.id, **ivs.iv_jumps()})
        if jump_rows:
            out = out.merge(pd.DataFrame(jump_rows), how='left', on='surface_id', validate='many_to_one')
        return out

    def get_feature_columns(self) -> List[str]:
        return ['iv_jump_theta', 'iv_jump_rho', 'iv_jump_psi']


class AddTenorTransformer(BaseTransformer):
    """
    Adds tenor feature to dataframe
    """
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = df.copy()
        v_tenor_eod = np.array([date_to_eod_expiry(dt) for dt in out['tenor_dt'].values])
        out['tenor'] = get_v_tenor(
            v_tenor_eod,
            np.array([x.astype('datetime64[us]').item() for x in out['ts_end'].values]),
        )
        return out

    def get_feature_columns(self) -> List[str]:
        return ['tenor']


class AddLaggedFeaturesTransformer(BaseTransformer):
    """
    Adds lagged features to dataframe
    """
    def __init__(self, max_lag_days: int | None = 20):
        self.max_lag_days = max_lag_days

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out: pd.DataFrame = df.copy()
        out = pivot_on_start_date(out, max_lag=self.max_lag_days)
        out = out.drop(
            [c for c in out.columns if 'iv_jump' in c and '_lag_' in c],
            axis=1,
            errors='ignore',
        )
        out = out.drop(
            [c for c in out.columns if 'tenor' in c and '_lag_' in c],
            axis=1,
            errors='ignore',
        )
        return out

    def get_feature_columns(self) -> List[str]:
        cols = [f'{metric}_lag_0' for metric in ['theta', 'rho', 'psi']]
        if self.max_lag_days:
            for metric in ['theta', 'rho', 'psi']:
                for i in range(1, self.max_lag_days):
                    cols.append(f'{metric}_lag_{i}')
        return cols


class AddRelativeDifferenceLagsTransformer(BaseTransformer):
    def __init__(self, max_lag_days: int | None = 20):
        self.max_lag_days = max_lag_days

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not (True or True):  # Since always included if in list
            return df
        out = df.copy()
        for metric in ['theta', 'rho', 'psi']:
            for i in range(1, self.max_lag_days or 0):
                lag_col = f'value_{metric}_lag_{i}'
                base_col = f'value_{metric}_lag_0'
                if lag_col in out.columns:
                    out[f'{metric}_rel_lag_{i}'] = out[lag_col] / out[base_col] - 1
                    out[f'{metric}_sub_lag_{i}'] = out[base_col] - out[lag_col]
        return out

    def get_feature_columns(self) -> List[str]:
        cols = []
        for metric in ['theta', 'rho', 'psi']:
            for i in range(1, self.max_lag_days or 0):
                cols.append(f'{metric}_rel_lag_{i}')
                cols.append(f'{metric}_sub_lag_{i}')
        return cols


class AddEarningsJumpLagsTransformer(BaseTransformer):
    """
    Add rate of change for theta/rho/psi by tenor for previous earnings jumps...
    """
    def __init__(self, max_er_lags: int = 4):
        self.max_er_lags = max_er_lags

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        - Extract all sym|release-date combinations from df
        - Get IVS for each on/after release date
        - Calc rate of change by tenor
        - Join to df. lag_er_{i}
        """
        out = df.copy()
        # Underlying | Lag | Tenor | Metric
        jump_dct = defaultdict(dict)

        for symbol, release_date in set(map(tuple, df[['underlying', 'release_date']].values)):
            equity = Equity(symbol)
            lag_release_dates = sorted([rd for rd in EarningsPreSessionDates(symbol) if rd <= release_date], reverse=True)

            for lag in range(1, self.max_er_lags + 1):
                if len(lag_release_dates) <= lag:
                    continue
                lag_release_date = lag_release_dates[lag]
                dt_pre_post = scoped_dates(lag_release_date, ScopePrePost.pm_1)

                if not IVSurface.is_cached(equity, [dt_pre_post[0]]) or not IVSurface.is_cached(equity, [dt_pre_post[1]]):
                    info(f'AddEarningsJumpLagsTransformer.transform(): No IVS for symbol {symbol} {release_date}')
                    continue

                v_ivs_pre = get_v_ivs(equity, dates=[dt_pre_post[0]], resolution=Resolution.second, seq_ret_threshold=0.002, arb_free=False, seq_ret_threshold_surface=None)
                v_ivs_post = get_v_ivs(equity, dates=[dt_pre_post[1]], resolution=Resolution.second, seq_ret_threshold=0.002, arb_free=False, seq_ret_threshold_surface=None)
                svi_params_pre = v_ivs_to_weighted_metrics(v_ivs_pre)
                svi_params_post = v_ivs_to_weighted_metrics(v_ivs_post)
                for tenor_dt, svi_params_pre in svi_params_pre.items():
                    if tenor_dt not in svi_params_post:
                        continue
                    tenor = (tenor_dt - lag_release_date).days / 365
                    jump_dct[(symbol, lag_release_date, SSVIMetric.theta)][tenor] = svi_params_post[tenor_dt].theta / svi_params_pre.theta
                    jump_dct[(symbol, lag_release_date, SSVIMetric.rho)][tenor] = svi_params_post[tenor_dt].rho / svi_params_pre.rho
                    jump_dct[(symbol, lag_release_date, SSVIMetric.psi)][tenor] = svi_params_post[tenor_dt].psi / svi_params_pre.psi

        # Assuming df has a 'tenor' column which corresponds to dt
        for index, row in out.iterrows():
            symbol = row['underlying']
            release_date = row['release_date']
            lag_release_dates = sorted([rd for rd in EarningsPreSessionDates(symbol) if rd <= release_date], reverse=True)
            dt = row['tenor_dt']
            for lag in range(1, self.max_er_lags + 1):
                if len(lag_release_dates) <= lag:
                    continue
                lag_release_date = lag_release_dates[lag]
                tenor = (dt - lag_release_date).days / 365
                for metric in [SSVIMetric.theta, SSVIMetric.rho, SSVIMetric.psi]:
                    key = (symbol, lag_release_date, metric)
                    if key in jump_dct:
                        keys = np.array(list(jump_dct[key].keys()))
                        ix_nearest = np.argmin(abs(keys - tenor))
                        if abs(keys[ix_nearest] / tenor - 1) > 0.2:
                            continue
                        val = jump_dct[key][keys[ix_nearest]]
                        col_name = f'{metric}_jump_rate_er-{lag}'
                        if col_name not in out.columns:
                            out[col_name] = None
                        out.loc[index, col_name] = val

        out = self.fill_required_cols(out)

        for c in out.columns:
            if 'jump_rate_er' in c:
                print(f'{c}: {out[c].notna().sum() / len(out):.2f}')

        return out

    def get_feature_columns(self) -> List[str]:
        cols = []
        for lag in range(1, self.max_er_lags + 1):
            cols.extend([f'theta_jump_rate_er-{lag}', f'rho_jump_rate_er-{lag}', f'psi_jump_rate_er-{lag}'])
        return cols


class AddSpotFingerprintTransformer(BaseTransformer):
    """
    Transformer that adds spot fingerprint features to the dataset.
    """
    def __init__(self, lookback_days: Tuple[int] = (20,)):
        self.lookback_days = lookback_days

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        client = Client()
        out = df.copy()

        # Get unique underlying and release_date combinations
        unique_combos = out[['underlying', 'release_date']].drop_duplicates()

        for lag in self.lookback_days:
            spot_features = {}
            for _, row in unique_combos.iterrows():
                underlying = row['underlying']
                release_date = row['release_date']

                # Load spot data for lookback_days before release_date
                start_date = release_date - timedelta(days=lag)
                try:
                    spot_data = client.history([Equity(underlying)], start_date, release_date, Resolution.daily, TickType.trade, SecurityType.equity)
                    prices = spot_data[str(Equity(underlying))]['close'].dropna()

                    if len(prices) >= 2:
                        # Calculate daily returns
                        daily_returns = prices.pct_change().dropna()

                        # Cumulative return
                        cum_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

                        # Annualized volatility (assuming 252 trading days)
                        ann_vol = daily_returns.std() * np.sqrt(252)

                        # Additional features: max drawdown, skewness
                        cumulative = (1 + daily_returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = drawdown.min()

                        skewness = daily_returns.skew()

                        spot_features[(underlying, release_date)] = {
                            'spot_cum_return_20d': cum_return,
                            'spot_ann_vol_20d': ann_vol,
                            'spot_max_drawdown_20d': max_drawdown,
                            'spot_skewness_20d': skewness,
                        }

                except Exception as e:
                    warning(e)

            # Merge the features back into df
            features_df = pd.DataFrame.from_dict(spot_features, orient='index', columns=[
                f'spot_cum_return_{lag}d', f'spot_ann_vol_{lag}d', f'spot_max_drawdown_{lag}d', f'spot_skewness_{lag}d'
            ]).reset_index()
            features_df.columns = ['underlying', 'release_date', f'spot_cum_return_{lag}d', f'spot_ann_vol_{lag}d', f'spot_max_drawdown_{lag}d', f'spot_skewness_{lag}d']

            out = out.merge(features_df, on=['underlying', 'release_date'], how='left')

        return out

    def get_feature_columns(self) -> List[str]:
        return ['spot_cum_return_20d', 'spot_ann_vol_20d', 'spot_max_drawdown_20d', 'spot_skewness_20d']


class AddVegaTransformer(BaseTransformer):
    """
    Computes per-row vega using the CUDA FD pricer and inserts it
    as a 'vega' column.  Must be applied BEFORE pivoting (i.e. while
    the frame still has one row per option data-point).

    Expected columns already present in the frame:
        spot, strike, tenor, iv, right (0/1 or OptionRight), ts_end (or calculation_date)
    """
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        df["vega"] = self._compute_vega(df)
        return df

    @staticmethod
    def _compute_vega(df: pd.DataFrame) -> pd.Series:
        from options.helper import date_to_sod, date_to_eod_expiry, get_v_tenor
        from options.types.enums import OptionRight

        result = np.full(len(df), np.nan, dtype=np.float64)

        # group by calculation date so we can batch the CUDA call
        calc_date_col = "calculation_date" if "calculation_date" in df.columns else "ts_end"
        df = df.copy()
        df["_calc_date"] = pd.to_datetime(df[calc_date_col]).dt.date

        for calc_date, grp in df.groupby("_calc_date"):
            try:
                equity_sym = grp["underlying"].iloc[0]
                equity = Equity(equity_sym)
                dividends = get_dividends(equity_sym, calc_date)
                yield_curve = YieldCurve().get_last_zero_curve(calc_date, equity)

                s = grp["spot"].values.astype(np.float64)
                k = grp["strike"].values.astype(np.float64)

                # tenor in years from calc_date to expiry
                expiry_arr = pd.to_datetime(grp["expiry_dt"]).dt.date.values \
                    if "expiry_dt" in grp.columns else \
                    pd.to_datetime(grp["tenor_dt"]).dt.date.values
                t = get_v_tenor(
                    np.array([date_to_eod_expiry(e) for e in expiry_arr]),
                    date_to_sod(calc_date),
                )
                rights = grp["right"].values
                v_is_call = np.array(
                    [1 if (r == OptionRight.call or r == 1) else 0 for r in rights],
                    dtype=np.int32,
                )
                iv = grp["iv"].values.astype(np.float64)

                vegas = get_vega_cuda(
                    s=s, k=k, t=t,
                    v_is_call=v_is_call,
                    iv=iv,
                    dividends=dividends,
                    calculation_date=calc_date,
                    yield_curve=yield_curve,
                    equity=equity,
                )
                result[grp.index] = vegas

            except Exception as e:
                info(f"AddVegaTransformer: failed for {calc_date}: {e}")
                # leave as NaN; downstream NA-drop will handle it

        return pd.Series(result, index=df.index)

    def get_feature_columns(self) -> List[str]:
        return []   # vega is a weight column, not a feature


class FinalizeFrameTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.rename(columns={c: c.replace('value_', '') for c in df.columns})
        return df
