import json
import os
import datetime
import pandas as pd
import numpy as np

from common.interfaces.iload_xy import ILoadXY
from shared.modules.logger import logger
from itertools import product
from functools import reduce
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from common.utils.window_aggregator import WindowAggregator
from common.paths import Paths
from connector.jl_tsdb_client import query, matching_metas
from layers.features.upsampler import Upsampler
from layers.predictions.event_extractor import EventExtractor
from common.utils.util_func import is_stationary

map_re_information2aggregator = {
    '^imbalance': ['sum'],
    '^volume': ['sum'],
    '^sequence': ['sum'],
    'bid_buy_size_imbalance_ratio': ['max', 'min', 'mean'],
    'bid_buy_count_imbalance_ratio': ['max', 'min', 'mean'],
    'bid_buy_size_imbalance_net': ['max', 'min', 'mean'],
    'bid_buy_count_imbalance_net': ['max', 'min', 'mean'],
}


class LoadXY(ILoadXY):
    """Estimate side by:
    - Loading label ranges
    - Samples are events where series diverges from expectation: load from disk client
    - Weights: Less unique sample -> lower weight
    - CV. Embargo area
    - Store in Experiment
    - Generate feature importance plot
    - Store: Signed estimates
    """

    def __init__(self, exchange: Exchange, sym, start: datetime, end: datetime, labels=None, signals=None, features=None, label_ewm_span='32min', from_pickle=False):
        self.exchange = exchange
        self.sym = sym
        self.start = start
        self.end = end
        self.labels = labels
        self.signals = signals
        self.features = features
        self.label_ewm_span = label_ewm_span
        self.window_aggregator_window = [int(2 ** i) for i in range(15)]
        self.dflt_window_aggregator_func = ['sum']
        self.window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, self.dflt_window_aggregator_func)]
        self.book_window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, ['sum'])]
        # self.book_window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, ['mean', 'max', 'min'])]
        self.boosters = []
        self.tags = {}
        self.df = None
        self.from_pickle = from_pickle

    def load_label(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        df_label = query(meta={
            'measurement_name': 'label', 'exchange': self.exchange, 'asset': self.sym,
            'col': 'forward_return_ewm',
            'ewm_span': self.label_ewm_span},
            start=self.start.date(),
            stop=self.end.date())
        df_label.columns = ['label']
        # this is too dangerously generalized. OUTER and FFILL in a row
        df_m = df.merge(df_label, how='outer', left_index=True, right_index=True).sort_index()
        df_m = self.curtail_nnan_front_end(df_m).ffill()
        df_m = df_m.iloc[df_m.isna().sum().max():]
        assert df_m.isna().sum().sum() == 0
        # return df_m[[c for c in df_m.columns if c != 'label']], df_m['label'] + 1  # multi class label
        return df_m[[c for c in df_m.columns if c != 'label']], df_m['label']  # return label

    @staticmethod
    def get_ix_nan_start_end(df):
        i_first_na = {}
        i_first_na_rev = {}
        for i in range(df.shape[1]):
            i_first_na[i] = np.argmax((~np.isnan(df.iloc[:, i].values)))
            i_first_na_rev[i] = np.argmax((~np.isnan(df.iloc[:, i].values[::-1])))
        for i, iloc_na in i_first_na.items():
            if iloc_na == 0 and np.isnan(df.iloc[0, i]):  # argmax returns 0 if all NAN or no maximum at all
                i_first_na_rev[i] = np.isnan(df.values[:, i]).sum()
        return i_first_na, i_first_na_rev

    def curtail_nnan_front_end(self, df):
        i_first_na, i_first_na_rev = self.get_ix_nan_start_end(df)
        iloc_begin = np.max(list(i_first_na.values()) or [0])
        iloc_end = len(df) - np.max(list(i_first_na_rev.values()) or [0])
        return df.iloc[iloc_begin:iloc_end]

    def apply_embargo(self):
        pass

    def calc_weights(self):
        pass

    def reduce_feature_frame(self, df: pd.DataFrame, skip_stationary=False) -> pd.DataFrame:
        df = self.exclude_non_stationary(df)
        df = self.exclude_too_little_data(df)  # this has an error in case of no NaNs
        # ffill after curtailing to avoid arriving at incorrect states for timeframes where information has simply not been loaded        df = self.curtail_nnan_front_end(df).ffill()
        # df = self.exclude_too_little_data(df)
        return df

    def load_by_meta(self, meta: dict, window_aggregator):
        metas = matching_metas(meta)
        dfs = []
        for meta in metas:
            df = query(meta, start=self.start.date(), stop=self.end.date())
            df = self.exclude_non_stationary(df)
            dfs += [self.sample_df(Upsampler(df[c]).upsample(aggregate_window.window, aggregate_window.aggregator)) for (c, aggregate_window) in product(df.columns, window_aggregator)]
        df = pd.concat(dfs, sort=True, axis=1)
        return self.reduce_feature_frame(df, skip_stationary=False)

    def load_features(self) -> pd.DataFrame:
        """order book imbalance, tick imbalance, sequence etc."""
        logger.info('Fetching Order book imbalances')
        df_book = self.load_by_meta({
            'measurement_name': 'order book', 'exchange': self.exchange,
            'delta_size_ratio': 0.5
            # 'asset': self.sym  # remove to get it for multiple syms....
        }, self.book_window_aggregators)
        ################################
        logger.info('Fetching Trade volume')
        df_trade_volume = self.load_by_meta({
            'measurement_name': 'trade bars',
            'exchange': self.exchange,
            # 'asset': self.sym,  # remove to get it for multiple syms....
            'information': 'volume'
        }, self.window_aggregators)
        #############################
        logger.info('Fetching trade imbalance')
        df_trade_imbalance = self.load_by_meta({
            'measurement_name': 'trade bars',
            'exchange': self.exchange,
            # 'asset': self.sym,  # remove to get it for multiple syms....
            'information': 'imbalance'
        }, self.window_aggregators)
        #######################
        logger.info('Fetching trade sequence')
        df_trade_sequence = self.load_by_meta({
            'measurement_name': 'trade bars',
            'exchange': self.exchange,
            # 'asset': self.sym,  # remove to get it for multiple syms....
            'information': 'sequence'
        }, self.window_aggregators)
        ##########
        logger.info('Trade book done')
        df = pd.concat((df_book, df_trade_imbalance, df_trade_sequence, df_trade_volume), sort=True, axis=1)
        df = self.exclude_too_little_data(df)
        df = self.fill_na_with_original_data(df)
        df = self.curtail_nnan_front_end(df)
        # df.ffill(inplace=True)
        # for c in df.columns:  # loop as calling on entire df caused mem error
        #     df[c].ffill(inplace=True)
        # df = self.reduce_feature_frame(df)
        df = self.exclude_too_little_data(df)
        # assert df.isna().sum().sum() == 0
        return df

    def join_right_ffill(self, df_left, meta, window_aggregator):
        ix_keep = df_left.index
        for m in matching_metas(meta):
            df = query(m, start=self.start.date(), stop=self.end.date())
            for ix, ps_right in df.iteritems():
                c = ps_right.name
                if c in df_left.columns:
                    logger.info(f'Join right ffil {c}')
                    ps_fill = ps_right[~ps_right.isna()]
                    ps_fill.name = 'join'
                    df_t = df_left[[c]].merge(ps_fill, how='outer', right_index=True, left_index=True, sort=True)
                    df_t['join'] = df_t['join'].ffill()
                    df_t[c] = df_t[c].fillna(df_t['join'])
                    df_left.loc[ix_keep, c] = df_t.loc[ix_keep, c]
            for ps_right in (Upsampler(df[c]).upsample(aggregate_window.window, aggregate_window.aggregator) for (c, aggregate_window) in product(df.columns, window_aggregator)):
                c = ps_right.name
                if c in df_left.columns:
                    logger.info(f'Join right ffil {c}')
                    ps_fill = ps_right[~ps_right.isna()]
                    ps_fill.name = 'join'
                    df_t = df_left[[c]].merge(ps_fill, how='outer', right_index=True, left_index=True, sort=True)
                    df_t['join'] = df_t['join'].ffill()
                    df_t[c] = df_t[c].fillna(df_t['join'])
                    df_left.loc[ix_keep, c] = df_t.loc[ix_keep, c]
        return df_left

    def fill_na_with_original_data(self, df):
        logger.info('Fetching Order book imbalances')
        # memory error here. need to derive only exactly the columns that's needed and only 1 col at a time...
        df = self.join_right_ffill(df, {
            'measurement_name': 'order book',
            'exchange': self.exchange,
            'delta_size_ratio': 0.5
        }, self.book_window_aggregators)
        ################################
        logger.info('Fetching Trade volume')
        df = self.join_right_ffill(df, {
            'measurement_name': 'trade bars',
            'exchange': self.exchange,
            'information': 'volume'
        }, self.window_aggregators)
        #############################
        logger.info('Fetching trade imbalance')
        df = self.join_right_ffill(df, {
            'measurement_name': 'trade bars',
            'exchange': self.exchange,
            'information': 'imbalance'
        }, self.window_aggregators)
        #######################
        logger.info('Fetching trade sequence')
        df = self.join_right_ffill(df, {
            'measurement_name': 'trade bars',
            'exchange': self.exchange,
            'information': 'sequence'
        }, self.window_aggregators)
        return df

    def exclude_too_little_data(self, df, pct_start=0.2, pct_end=0.2) -> pd.DataFrame:
        """This needs to be replace with exclude too little variance ..."""
        if len(df) == 0:
            return df
        col_exclude = []
        i_first_na, i_first_na_rev = self.get_ix_nan_start_end(df)
        iloc_begin = np.array(list(i_first_na.values()))
        iloc_end = np.array(list(i_first_na_rev.values()))
        # exclude cols that
        #   have too large iloc_end or iloc begin or
        #   variance is too small
        for i in range(df.shape[1]):
            if i_first_na[i] > len(df) * pct_start:
                col_exclude.append(i)
            if i_first_na_rev[i] > len(df) * pct_end:
                col_exclude.append(i)
        # mean_range = (iloc_end - iloc_begin).mean() / 2
        # col_range = dict(zip(list(df.columns), (iloc_end - iloc_begin) > mean_range))

        # ps_cnt_nonna = df.isna().sum().to_dict()
        # mean_cnt_non_na = np.mean(list(ps_cnt_nonna.values()))
        # ex_cols = r"\n".join([c for c, cnt in ps_cnt_nonna.items() if cnt > mean_cnt_non_na])
        # logger.info(f'Removing columns containing > {mean_range} NANs`: {[c for c in col_range.keys() if not col_range[c]]}')
        col_exclude = list(set(col_exclude))
        logger.info(f'exclude_too_little_data: Cols with too few values start/end values: {[df.columns[i] for i in col_exclude]}')
        return df.iloc[:, [i for i in range(len(df.columns)) if i not in col_exclude]]

    @staticmethod
    def exclude_non_stationary(df) -> pd.DataFrame:
        with open(os.path.join(Paths.config, 'stationary.json'), 'r') as f:
            col_stationary = json.load(f)
        with open(os.path.join(Paths.config, 'non_stationary.json'), 'r') as f:
            non_stationary = json.load(f)
        logger.info(f'Not Stationary - Excluding {[c for c in df.columns if c in non_stationary]}')
        df = df[[c for c in df.columns if c not in non_stationary]]

        def f_is_ps_stat(ps: pd.Series):
            res = is_stationary(ps[ps.notna()].values)
            if not res:
                logger.warning(f'{ps.name} is not stationary. Excluding!')
            return res

        col_stationary += [c for c in df.columns if f_is_ps_stat(df[c]) if c not in col_stationary]
        if len(df) > 1_000_000:
            with open(os.path.join(Paths.config, 'stationary.json'), 'wt') as f:
                json.dump(sorted(set(col_stationary + col_stationary)), f)

            non_stationary += sorted(set([c for c in df.columns if c not in col_stationary]))
            with open(os.path.join(Paths.config, 'non_stationary.json'), 'wt') as f:
                json.dump(non_stationary, f)
        return df[[c for c in col_stationary if c in df.columns]]

    def exclude_low_variance_cols(self, df):
        variances = df.var()
        df = df.drop(variances[variances < variances.median() / 2].index.tolist(), axis=1)
        return df

    @staticmethod
    def sample_df(df_in: pd.DataFrame | pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(df_in, columns=[df_in.name]) if isinstance(df_in, pd.Series) else df_in.copy()
        ts_sample = pd.Index(reduce(lambda res, item: res.union(set(item)), (EventExtractor(thresholds=[3, 3])(df[col], len(df) / 10) for col in df.columns), set()))
        return df.loc[ts_sample].sort_index()

    def assemble_frames(self):
        self.df = self.load_features()
        self.df = self.exclude_low_variance_cols(self.df)
        self.df, self.ps_label = self.load_label(self.df)

        # ts_sample = pd.Index(reduce(lambda res, item: res.union(set(item)), (EventExtractor(thresholds=[5, 5])(self.df[col], len(self.df) / 10) for col in self.df.columns), set()))
        ts_sample = self.df.index
        logger.info(f'len ts_sample: {len(ts_sample)} - {100 * len(ts_sample) / len(self.df)}% of df')
        if not ts_sample.empty:
            ix_sample = ts_sample.intersection(self.ps_label.index)
            self.df, self.ps_label = self.df.loc[ix_sample].sort_index(), self.ps_label.loc[ix_sample].sort_index()
        logger.info(f'DF Shape: {self.df.shape}')


if __name__ == '__main__':
    exchange = Exchange.bitfinex
    sym = Assets.ethusd

    inst = LoadXY(
        exchange=exchange,
        sym=sym,
        start=datetime.datetime(2022, 2, 7),
        # end=datetime.datetime(2022, 2, 8),
        end=datetime.datetime(2022, 3, 13),
    )
    inst.assemble_frames()
