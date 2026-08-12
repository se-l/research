import os
import lightgbm as lgb
import numpy as np

from typing import List, Tuple, Dict
from joblib import load
from numpy import mean

from sklearn.base import BaseEstimator, RegressorMixin
from options.types.iv_surface_essvi import MetricSSVI
from shared.modules.logger import info
from shared.paths import Paths


class ESSVITreeEstimator(BaseEstimator, RegressorMixin):
    model_base_name = 'earnings_iv_drop_tree_regressors'

    def __init__(self, metric: MetricSSVI | str):
        self.metric = metric
        self.y_col_nm = metric
        self.models: List = []
        self.best_iteration = None

    @staticmethod
    def transform(x):
        return x

    def fit(self, X, y, v_ix_cv: List[Tuple[np.ndarray, np.ndarray]], alpha: float = None, weight: np.ndarray = None, train_params: dict = None):
        if weight is not None:
            assert len(weight) == len(y) == len(X), f'len(weight)={len(weight)} != len(y)={len(y)} != len(X)={len(X)}'

        scores = []
        params = {}
        for i, (ix_train, ix_cv) in enumerate(v_ix_cv):

            X_train, X_cv = X.iloc[ix_train], X.iloc[ix_cv]
            y_train, y_cv = y.iloc[ix_train], y.iloc[ix_cv]
            if weight is not None:
                weight_train, weight_cv = weight[ix_train], weight[ix_cv]
            else:
                weight_train, weight_cv = None, None

            params = train_params or self.params_dflt(alpha)

            lgb_train = lgb.Dataset(X_train, y_train, weight=weight_train)
            lgb_eval = lgb.Dataset(X_cv, y_cv, weight=weight_cv, reference=lgb_train)

            gbm = lgb.train(params, lgb_train, valid_sets=[lgb_eval])

            self.models.append(gbm)
            scores.append(gbm.best_score['valid_0'])

        cv_mean_score = mean(list([list(el.values())[0] for el in scores]))
        info(f'CV {params["metric"]} mean: {cv_mean_score}')

        return params["metric"].upper(), cv_mean_score

    def params_dflt(self, alpha: float = None) -> Dict:
        dct = {
            'objective': 'quantile' if alpha else 'regression',
            'metric': 'quantile' if alpha else 'rmse',
            'min_data_in_bin': 3,
            'min_data_in_leaf': 5,
            # 'num_leaves': 31,
            # 'learning_rate': 0.01,
            # 'feature_fraction': 0.9,
            # 'bagging_fraction': 0.8,
            # 'bagging_freq': 5,
            # 'verbose': 0
        }
        if alpha:
            dct['alpha'] = alpha
        return dct

    def rmse(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def predict(self, X):
        res = []
        for m in self.models:
            res.append(m.predict(X[self.feature_names(m)], num_iteration=m.best_iteration))
        return np.mean(res, axis=0)

    def feature_names(self, model=None):
        return (model or self.models[0]).feature_name()

    @classmethod
    def model_name(cls, version: str, alpha: float = 0.5):
        _alpha = alpha or 0.5
        return f'{cls.model_base_name}_{version}_{_alpha:.2f}.joblib'

    @classmethod
    def load_model(cls, version: str, alpha: float = 0.5):
        _alpha = alpha or 0.5
        return load(os.path.join(Paths.path_models, cls.model_name(version, _alpha)))


if __name__ == '__main__':
    earnings_iv_drop_regressor_model_name_version = 'f_20260516-230556'
    alpha = None
    m = ESSVITreeEstimator.load_model(earnings_iv_drop_regressor_model_name_version, 0.5)
    # now store all boosters to disk
    dir_boosters = Paths.path_models.joinpath(f'{earnings_iv_drop_regressor_model_name_version}-{alpha:.2f}')
    if not os.path.exists(dir_boosters):
        os.mkdir(dir_boosters)
    for metric, inst in m.items():
        if not os.path.exists(dir_boosters.joinpath(metric)):
            os.mkdir(dir_boosters.joinpath(metric))
        for i, booster in enumerate(inst.models):
            path_booster = dir_boosters.joinpath(metric, f"model_{i}.txt")
            booster.save_model(path_booster)
            print(f'Stored {path_booster}')


