import os
import pickle
import json5
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from typing import List, Dict
from datetime import timedelta, date
from scipy.optimize import curve_fit
from dataclasses import dataclass, asdict
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import r2_score

from options.helper import year_quarter, tenor, atm_iv, enrich_atm_iv
from options.typess.enums import Resolution
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from shared.constants import EarningsPreSessionDates
from shared.paths import Paths
from shared.plotting import show


def a_bx_cx2_dx3(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


@dataclass
class ModelEarningsIVDropRegressor:
    p: float
    min_tenor: float
    result: Dict[str, List[float]]


class EarningsIVDropRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, p=0.3, min_tenor=0.1, f=a_bx_cx2_dx3):
        self.p = p
        self.min_tenor = min_tenor
        self.f = f
        self.result = {}

    def fit_transform(self, x, y):
        self.x = x.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

        self.result = {}
        for right, s_df in self.x[self.x['tenor'] > self.min_tenor].groupby('right'):
            vx = np.power(s_df['tenor'], p)
            popt, pcov = curve_fit(self.f, vx.values, self.y.loc[s_df.index].values)
            self.result[right] = list(popt)
        return self

    def score(self, x, y, sample_weight=None):
        # print(f'{sym} {right} p: {p}, r2: {r2}, popt: {popt}')
        return r2_score(y, self.predict(x))

    def predict(self, x: pd.DataFrame) -> np.array:
        drop = False
        if 'right' not in x.columns:
            drop = True
            x['right'] = x.index.get_level_values('right').values
        val = x.apply(self.predict_row, axis=1).values
        if drop:
            x.drop(columns='right', inplace=True)
        return val

    def predict_row(self, ps: pd.Series) -> float:
        vx = np.power(ps['tenor'], self.p)
        return self.f(vx, *self.result[ps['right']])

    def store_estimator(self, path: str):
        """With training data. Will fail to load when estimator code/signature changes"""
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return self

    def store_model(self, path):
        """Without training data, just for prediction."""
        obj = ModelEarningsIVDropRegressor(self.p, self.min_tenor, self.result)
        with open(path, 'w') as fp:
            json5.dump(asdict(obj), fp)
        return self

    def load_model(self, path):
        with open(path, 'r') as fp:
            obj = json5.load(fp)
        self.p = obj['p']
        self.min_tenor = obj['min_tenor']
        self.result = obj['result']
        return self


if __name__ == "__main__":
    plot = True
    # from sklearn.utils.estimator_checks import check_estimator
    # check_estimator(EarningsIVDropRegressor())
    tickers = 'DG,ORCL,ONON,DLTR,CSCO,CRWD,PATH,DELL,TGT,JD,WDAY,PANW,MRVL'
    tickers = 'DG,ORCL,ONON,DLTR,CSCO,CRWD,PATH,TGT,JD,WDAY,PANW,MRVL'
    # tickers = 'TGT,DG'
    v_xy = []

    level_groupby = ('expiry', 'right')
    for take in [-1, -2]:
    # for take in [-1]:
        for sym in tickers.split(','):
            sym = sym.lower()
            equity = Equity(sym)
            # start, end = earnings_download_dates(sym, take=take)
            resolution = Resolution.minute
            seq_ret_threshold = 0.005
            release_date = EarningsPreSessionDates(sym)[take]
            option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, year_quarter(release_date))
            df = option_frame.df_options.sort_index()

            ts = df.index.get_level_values('ts').unique()
            v_ts_pre_release = [i for i in ts if i.date() == release_date if i.hour >= 10]
            v_ts_post_release = [i for i in ts if i.date() == (release_date + timedelta(days=1)) if i.hour >= 10]

            x0 = df.loc[v_ts_pre_release]
            x1 = df.loc[v_ts_post_release]
            for df_ in [x0, x1]:
                enrich_atm_iv(df_)

            agg_funcs = dict(atm_iv='mean')
            x0 = x0.groupby(level=level_groupby).agg(agg_funcs)
            x1 = x1.groupby(level=level_groupby).agg(agg_funcs)

            ix_intersect = x0.index.intersection(x1.index)
            x0 = x0.loc[ix_intersect]
            x1 = x1.loc[ix_intersect]

            x = x0.merge(x1['atm_iv'], left_index=True, right_index=True, suffixes=('', '_1'))
            x['expiry'] = x.index.get_level_values('expiry').values
            x['tenor'] = x['expiry'].apply(lambda expiry: tenor(expiry, release_date))
            x.drop(columns='expiry', inplace=True)

            x['d_atm_iv'] = x['atm_iv_1'] - x['atm_iv']
            x['d_atm_iv_pc'] = x['d_atm_iv'] / x['atm_iv']

            x = x.reset_index().drop(columns=['expiry', 'atm_iv', 'atm_iv_1']).dropna()
            x['symbol'] = sym
            v_xy.append(x)

    x = pd.concat(v_xy)
    if plot:
        def a_bpx(x, a, b):
            return a + b**x
        def a_bx_cx2(x, a, b, c):
            return a + b * x + c * x ** 2
        def a_bx_cx2_dx3(x, a, b, c, d):
            return a + b * x + c * x ** 2 + d * x ** 3

        powers = [1, 0.3, 0.35, 0.4]
        # use 0.3
        f = a_bx_cx2_dx3
        fig = make_subplots(rows=4, cols=1)
        for i, p in enumerate(powers):
            row = i + 1
            t1m = np.power(30 / 365, p)

            for right, s_df in x[x['tenor'] > 0.1].groupby('right'):
                vx = np.power(s_df['tenor'], p)
                vy = s_df['d_atm_iv_pc']
                # sigma = 1/vx
                popt, pcov = curve_fit(f, vx.values, vy.values)
                y_pred = f(vx, *popt)
                r2 = r2_score(vy, y_pred)
                print(f'{sym} {right} p: {p}, r2: {r2}, popt: {popt}')
                fig.add_trace(go.Scatter(x=vx, y=y_pred, mode='markers', marker=dict(size=6), name=f'PRED {p} {sym} {right}'), row=row, col=1)

                for sym, ss_df in s_df.groupby('symbol'):
                    vx = np.power(ss_df['tenor'], p)
                    vy = ss_df['d_atm_iv_pc']
                    fig.add_trace(go.Scatter(x=vx, y=vy, mode='markers', marker=dict(size=3), name=f'{p} {sym} {right}'), row=row, col=1)

            fig.add_vline(x=t1m, line_dash='dash', line_color='black', row=row, col=1)
        show(fig)
    # fit puts and calls separately. puts seem to drop more than calls.
    # Best still would be to somehow previously differentiate diffusion IV and jump IV. Then subtract jump IV and fit the remainder...
    # explore more x, y transforms straightening the line. better transform + linear regression = better fit rather than curve fit.
    # consider calibrating which power law gives the straigtest line after linear regression
    y = x.pop('d_atm_iv_pc')
    path_model = os.path.join(Paths.path_models, f'earnings_iv_drop_regressor_{date.today()}b.json')
    model = EarningsIVDropRegressor(p=0.3).fit_transform(x, y).store_model(path_model)
