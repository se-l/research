import pickle
import os
import datetime
import re
import numpy as np
import pandas as pd
import lightgbm as lgb

from shared.modules.logger import logger
from connector.tsdb.client import query
from layers.features.upsampler import Upsampler
from typing import List
from functools import reduce
from common.utils.util_func import is_stationary
from sandbox.influxdb.influxdb_wrapper import WindowAggregator
from common.paths import Paths


class Predict:
    def __init__(self, models: [lgb.Booster], start: datetime.datetime, end: datetime.datetime):
        self.models = models
        self.start = start
        self.end = end

    def all_features(self) -> List[str]:
        return list(reduce(lambda res, model: res.union(set(model.feature_name())), self.models, set()))

    @staticmethod
    def name2tags(name: str) -> dict:
        res = {}
        for item in name.split('|'):
            if item.startswith(('aggWindow', 'aggAggregator', 'levels', 'unit_size', 'alpha')):
                continue
            else:
                key, value = item.split('-')
                value = value.replace('order_book', 'order book').replace('trade_bars', 'trade bars')
                res[key] = value
        return res

    @staticmethod
    def name2window_aggregate(name: str):
        aggWindow, aggAggregator = None, None
        match = re.search(r'aggWindow-(\d*)', name)
        if match:
            aggWindow = int(match.group(1).split('|')[0])
        match = re.search(r'aggAggregator-(\w*)', name)
        if match:
            aggAggregator = match.group(1).split('|')[0]

        if aggWindow and aggAggregator:
            return WindowAggregator(aggWindow, aggAggregator)

    def predict(self):
        feats = self.all_features()
        pss = []
        if False:
            for c in feats:
                logger.info(f'Loading {c}')
                try:
                    df = pd.DataFrame(query(meta=self.name2tags(c), start=self.start, to=self.end)).set_index(0)
                except Exception as e:
                    logger.info(e)
                    logger.info(c)
                    continue
                window_agg = self.name2window_aggregate(c)
                pss.append(Upsampler(df.iloc[:, 0]).upsample(window_agg.window, window_agg.aggregator))
            df = pd.concat(pss, axis=1, sort=True)

            with open(os.path.join(Paths.data, 'dfpredict.p'), 'wb') as f:
                pickle.dump(df, f)
        else:
            with open(os.path.join(Paths.data, 'dfpredict.p'), 'rb') as f:
                df = pickle.load(f)
        preds = []
        for m in self.models:
            preds.append(m.predict(df))
        self.preds = pd.DataFrame(np.array(preds).mean(axis=0), index=df.index)
        self.preds.columns = ['short', 'flat', 'long']
        return self.preds

    def __call__(self, ps: pd.Series) -> pd.Index:
        if ps.isna().sum() > 0 or not is_stationary(ps.values):
            return pd.Index([])
        elif self.method == 'std':
            threshold = ps.std()
            return ps.index[ps.abs() >= threshold]
        else:
            raise NotImplementedError('Unclear method how to establish a range')


if __name__ == '__main__':
    ex = 'ex2022-03-06_165641-ethusd'
    with open(os.path.join(Paths.trade_model, ex, 'boosters.p'), 'rb') as f:
        boosters = pickle.load(f)
    start = datetime.datetime(2022, 2, 7)
    end = datetime.datetime(2022, 3, 13)
    f1_ho = Predict(boosters, start, end).predict()
    print(f1_ho)
    with open(os.path.join(Paths.trade_model, ex, 'f1_ho.p'), 'wb') as f:
        pickle.dump(f1_ho, f)
