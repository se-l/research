import datetime
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import Counter, defaultdict
from common.interfaces.iload_xy import ILoadXY
from shared.modules.logger import logger
from itertools import product
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from common.utils.util_func import ex
from common.utils.window_aggregator import WindowAggregator
from connector.jl_tsdb_client import upsert
from layers.predictions.estimate.base import EstimateBase
from layers.predictions.load_xy import LoadXY
from layers.predictions.sample_weights import SampleWeights


class EstimateQuantiles(EstimateBase):  # inherit some interface
    """Estimate side by:
    - Loading label ranges
    - Samples are events where series diverges from expectation
    - Weights: Less unique sample -> lower weight
    - CV. Embargo area
    - Store in Experiment
    - Generate feature importance plot
    - Store: Signed estimates
    """

    def __init__(self, sym: Assets, load_xy=None, quantiles=None):
        self.load_xy: ILoadXY = load_xy
        # self.quantiles = quantiles if quantiles else [0.1, 0.5, 0.9]
        self.quantiles = quantiles if quantiles else [0.5]
        self.window_aggregator_window = [int(2**i) for i in range(13)]
        self.window_aggregator_func = ['sum']
        self.window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, self.window_aggregator_func)]
        self.tags = {}
        self.sym = sym
        self.ex = ex(sym)
        logger.info(self.ex)
        self.df = None
        self.boosters = defaultdict(list)
        self.pred_label_val = {}
        self.pred_label_ho = {}

    def load_inputs(self, features=None):
        self.load_xy.assemble_frames()
        logger.info('Load XY done.')
        # apply sampling and purging separately perhaps
        self.df = self.load_xy.df
        if features:
            self.df = self.df[[f.replace('order_book', 'order book').replace('trade_bars', 'trade bars') for f in features]]
        self.ps_label = self.load_xy.ps_label
        self.ps_label_return = self.load_xy.ps_label
        # ix = np.where((self.df.index > datetime.datetime(2022, 2, 14, tzinfo=pytz.UTC)) & (self.df.index < datetime.datetime(2022, 3, 1, tzinfo=pytz.UTC)))[0]
        # self.df = self.df.iloc[ix]
        # self.ps_label = self.ps_label.iloc[ix]

    @staticmethod
    def purge_overlap(train_index, test_index, i=250):
        """
        In each iteration remove i periods from around test.
        i should be derived somewhat intelligently ...
        """
        if test_index[0] == 0:  # test on left side
            return train_index[i:]
        elif test_index[0] > train_index[-1]:  # test on right side
            return train_index[:-i]
        else:
            i_begin = test_index[0] - 1
            i_end = test_index[-1] + 1
            return np.array(train_index[:(train_index.tolist().index(i_begin) - i)].tolist() + train_index[(train_index.tolist().index(i_end) + i):].tolist())

    def split_ho(self, ho_share=0.3):
        n_ho = int(len(self.df) * ho_share // 1)
        n_cv = len(self.df) - n_ho
        df = self.df.iloc[:n_cv]
        ps_label = self.ps_label.iloc[:n_cv]
        df_ho = self.df.iloc[-n_ho:]
        ps_label_ho = self.ps_label.iloc[-n_ho:]
        self.df = df
        self.ps_label = ps_label
        self.df_ho = df_ho
        self.ps_label_ho = ps_label_ho

    def store_return_attribution(self, ps: pd.Series):
        meta = {
            "measurement_name": "weights",
            "asset": self.sym,
            "information": "return_attribution_sample_weights",
            "ex": self.ex,
        }
        upsert(meta, ps)

    def train(self):
        self.split_ho()
        kf = KFold(n_splits=5, shuffle=False)

        arr_weight = SampleWeights(ps_label=self.ps_label, df=self.df).\
            return_attribution_sample_weights()
            # geometric_mean()
            # cluster_sample_weight(50).\
        arr_weight = arr_weight.weight_arrays['return_attribution_sample_weights']

        # self.store_return_attribution(pd.Series(
        #     arr_weight,  #.weight_arrays['return_attribution_sample_weights'],
        #     index=self.ps_label.index
        # ))

        quantile_booster_scores = defaultdict(list)
        pred_label_val = defaultdict(list)
        pred_label_ho = defaultdict(list)
        for quantile in self.quantiles:
            logger.info(f'Quantile {quantile}')
            estimator_params = {
                'objective': 'mse',
                # 'objective': 'quantile',
                # 'alpha': quantile,
                # 'verbosity': 0,
                'learning_rate': 0.05,
                'early_stopping_round': 100,
                'boosting_type': 'gbdt',
                'num_iterations': 1000,
                'device': 'gpu'
            }
            preds_val = []
            preds_ho = []
            for train_index, test_index in kf.split(self.df.index):
                train_index = self.purge_overlap(train_index, test_index)
                x_train, x_test = self.df.iloc[train_index], self.df.iloc[test_index]
                y_train, y_test = self.ps_label.iloc[train_index], self.ps_label.iloc[test_index]
                dataset_train = lgb.Dataset(x_train, label=y_train, weight=arr_weight[train_index])
                lgb_booster = lgb.train(estimator_params,
                                        train_set=dataset_train,
                                        valid_sets=[lgb.Dataset(x_test, label=y_test, weight=arr_weight[test_index]), dataset_train],
                                        valid_names=['valid_0', 'valid_train'],
                                        )
                self.boosters[quantile].append(lgb_booster)
                quantile_booster_scores[quantile].append(lgb_booster.best_score['valid_0'])
                preds_val.append(pd.DataFrame(lgb_booster.predict(self.df.iloc[test_index]), index=self.df.iloc[test_index].index))
                preds_ho.append(pd.DataFrame(lgb_booster.predict(self.df_ho), index=self.df_ho.index))

            preds_val = pd.concat(preds_val, axis=0).sort_index().groupby(level=0).mean()
            preds_ho = pd.concat(preds_ho, axis=0).sort_index().groupby(level=0).mean()
            self.pred_label_val[quantile] = preds_val.merge(self.ps_label, how='inner', right_index=True, left_index=True)
            self.pred_label_ho[quantile] = preds_ho.merge(self.ps_label_ho, how='inner', right_index=True, left_index=True)

    def evaluate_quantiles(self, estimator_params, pred_label_val, pred_label_ho):
        ho_quantile_mae = {}
        ho_quantile_rmse = {}
        quantiles = list(range(1, 10))
        for quantile in quantiles:
            threshold = pred_label_ho['label'].quantile(quantile/10)
            if threshold >= 1:
                ix = np.where(pred_label_ho.iloc[:, 0] > threshold)[0]
            else:
                ix = np.where(pred_label_ho.iloc[:, 0] < threshold)[0]
            if len(ix) == 0:
                ho_quantile_mae[quantile] = ho_quantile_rmse[quantile] = None
            else:
                ho_quantile_mae[quantile] = mean_absolute_error(pred_label_ho.values[ix, 0], pred_label_ho.values[ix, 1])
                ho_quantile_rmse[quantile] = mean_squared_error(pred_label_ho.values[ix, 0], pred_label_ho.values[ix, 1])
        plt.plot(quantiles, ho_quantile_mae.values())
        plt.xlabel('Quantile MAE, RMSE of K')
        plt.ylabel('MAE')
        plt.title('Quantile')
        plt.show()

        ho_quantile_mae = {}
        ho_quantile_rmse = {}
        quantiles = list(range(1, 10))
        for quantile in quantiles:
            threshold = pred_label_ho['label'].quantile(quantile / 10)
            if threshold >= 1:
                ix = np.where(pred_label_ho.iloc[:, 0] > threshold)[0]
            else:
                ix = np.where(pred_label_ho.iloc[:, 0] < threshold)[0]
            if len(ix) == 0:
                ho_quantile_mae[quantile] = ho_quantile_rmse[quantile] = None
            else:
                ho_quantile_mae[quantile] = mean_absolute_error(np.ones(len(ix)), pred_label_ho.values[ix, 1])
                ho_quantile_rmse[quantile] = mean_squared_error(np.ones(len(ix)), pred_label_ho.values[ix, 1])
        plt.plot(quantiles, ho_quantile_mae.values())
        plt.xlabel('Quantile MAE guessing return == 1')
        plt.ylabel('MAE')
        plt.title('Quantile')
        plt.show()

    def best_k_elbow(self, k_max: int):
        logger.info('Find optimal # k cluster using Elbow method')
        sum_squared_distances = []
        k = list(range(2, k_max))
        for i, num_clusters in enumerate(k):
            kmeans = MiniBatchKMeans(n_clusters=num_clusters,
                                 # random_state=0,
                                 # batch_size=6,
                                 max_iter=1000).fit(self.df.values)
            sum_squared_distances.append(kmeans.inertia_)
        res = pd.Series(dict(zip(k, sum_squared_distances))).plot()
        plt.xlabel('Values of K')
        plt.ylabel('Sum of squared distances/Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def best_k_silhouette(self, k_max: int, k_min: int = 2):
        logger.info('Find optimal # k cluster using Silhouette score')

        def silhouette_score(values, cluster_labels):
            scores = []
            map_label2vec_internal = {label: values[np.where(cluster_labels == label)[0], :] for label in np.unique(cluster_labels)}
            map_label2vec_external = {label: values[np.where(cluster_labels != label)[0], :] for label in np.unique(cluster_labels)}
            for i, label in enumerate(cluster_labels):
                internal_distance = np.linalg.norm(map_label2vec_internal[label] - values[i], axis=1).mean()
                external_distance = np.linalg.norm(map_label2vec_external[label] - values[i], axis=1).mean()
                scores.append((external_distance - internal_distance) / max(internal_distance, external_distance))
            return np.mean(scores)
        k = list(range(k_min, k_max))
        silhouette_avg = []
        for i, num_clusters in enumerate(k):
            if i % 10 == 0:
                print(i)
            # initialise kmeans
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000).fit(self.df.values)
            # silhouette score
            silhouette_avg.append(silhouette_score(self.df.values, kmeans.labels_))
        plt.plot(k, silhouette_avg)
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette analysis For Optimal k')
        plt.show()
        # Plot Dispersity
        dct = dict(Counter(kmeans.labels_))
        tup_lst = [(k, v) for k, v in dct.items()]
        tup_lst = sorted(tup_lst, key=lambda tup: tup[1], reverse=True)
        plt.bar(list(range(len(tup_lst))), [tup[1] for tup in tup_lst])
        plt.xlabel('Cluster Label')
        plt.ylabel('Count states')
        plt.title('Points per cluster')
        plt.show()

        return k[silhouette_avg.index(pd.Series(silhouette_avg).bfill().ffill().min())]


if __name__ == '__main__':
    exchange = Exchange.bitfinex
    sym = Assets.ethusd
    start = datetime.datetime(2022, 2, 7)
    end = datetime.datetime(2022, 2, 13)
    # end = datetime.datetime(2022, 9, 3)
    logger.info(f'From {start} to {end}')
    inst = EstimateQuantiles(
        sym=sym,
        load_xy=LoadXY(exchange=exchange, sym=sym, start=start, end=end, labels=None, signals=None, features=None,
                       label_ewm_span='64min')
    )
    # features = ['_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-1500|aggWindow-256|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-256|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-1500|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-512|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-1024|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-min', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-150000|aggWindow-2048|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-2048|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-15|aggWindow-128|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-min', '_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-32|aggAggregator-max', '_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean', '_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-max', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-min', '_measurement-trade_bars|_field-imbalance_size|asset-solusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-4000|aggWindow-2048|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-128|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-1024|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-min', '_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean', '_measurement-trade_bars|_field-imbalance_size|asset-solusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-4000|aggWindow-256|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min', '_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-7000|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min', '_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-btcusd|unit_size-5|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-max', '_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-max', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-256|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean', '_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-16|aggAggregator-min', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-128|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-2048|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-2048|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-solusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-4000|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-256|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean', '_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean', '_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-2048|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-2048|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean', '_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-2048|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-xrpusd|unit_size-10000|aggWindow-2048|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-256|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-btcusd|unit_size-5|aggWindow-2048|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-512|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-btcusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-300000|aggWindow-128|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-mean', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-128|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-32|aggAggregator-max', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-512|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-ethusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-75000|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-max', '_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-max', '_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-min', '_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-btcusd|unit_size-5|aggWindow-1024|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-max', '_measurement-trade_bars|_field-imbalance_size|asset-ethusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-30|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-min', '_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-256|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-15|aggWindow-512|aggAggregator-sum', '_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-256|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-256|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-min', '_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-min', '_measurement-order_book|_field-count_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean', '_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean', '_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean', '_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean', '_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-mean', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean', '_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-xrpusd|unit_size-10000|aggWindow-256|aggAggregator-sum', '_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-min', '_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-8|aggAggregator-min', '_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-max', '_measurement-trade_bars|_field-sequence_direction|asset-btcusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-300000|aggWindow-1024|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-xrpusd|unit_size-10000|aggWindow-512|aggAggregator-sum', '_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-512|aggAggregator-sum', '_measurement-trade_bars|_field-imbalance_size|asset-ethusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-30|aggWindow-2048|aggAggregator-sum']
    features = None
    inst.load_inputs(features=features)
    # inst.best_k_elbow(100)
    # inst.best_k_silhouette(50, 50)
    # k ==50 is okay. rather have more than fewer. only useful if actually disperse. means some k have
    # much larger count than others... need dispersity measure for each k like median cnt?
    inst.train()
    inst.save()
    # inst.to_disk()
    logger.info(f'Done. Ex: {inst.ex}')
