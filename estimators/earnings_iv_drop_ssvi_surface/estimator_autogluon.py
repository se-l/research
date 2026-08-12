import os
from uuid import uuid4
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from autogluon.common.utils.utils import setup_outputdir

from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, RegressorMixin

from options.types.iv_surface_essvi import MetricSSVI
from shared.modules.logger import info
from shared.paths import Paths


class ESSVITreeEstimator(BaseEstimator, RegressorMixin):
    """
    AutoGluon-based replacement for ESSVITreeEstimator.

    Multi-target wrapper over 3 TabularPredictors:
      - theta_y
      - rho_y
      - psi_y

    Interface is kept similar to the LightGBM estimator, but:
      - alpha / quantile regression is ignored for now
      - metric may be None (preferred), or one of theta_y/rho_y/psi_y for compatibility
    """
    model_base_name = 'earnings_iv_drop_autogluon_regressors'
    target_cols = ['theta_y', 'rho_y', 'psi_y']

    def __init__(self, metric: MetricSSVI | str | None = None):
        self.metric = metric
        self.y_col_nm = metric
        self.models: List[Dict[str, str]] = []   # fold -> {target: predictor_path}
        self.final_model: Dict[str, str] = {}    # {target: predictor_path}
        self._predictor_cache: Dict[str, TabularPredictor] = {}

    @staticmethod
    def transform(x):
        return x

    def fit(
        self,
        X,
        y,
        v_ix_cv: List[Tuple[np.ndarray, np.ndarray]],
        alpha: float = None,
        weight: np.ndarray = None,
        train_params: dict = None
    ):
        """
        Expects:
          X: pd.DataFrame
          y: pd.DataFrame with columns ['theta_y', 'rho_y', 'psi_y']
        """
        if alpha is not None:
            info('AutoGluon estimator: alpha argument ignored for now; using standard regression.')

        if weight is not None:
            assert len(weight) == len(X) == len(y), f'len(weight)={len(weight)} != len(X)={len(X)} != len(y)={len(y)}'

        assert isinstance(X, pd.DataFrame), 'X must be a pandas DataFrame'
        assert isinstance(y, pd.DataFrame), 'y must be a pandas DataFrame'
        assert all(c in y.columns for c in self.target_cols), f'y must contain {self.target_cols}'

        params = train_params or self.params_dflt()

        scores_by_target = {target: [] for target in self.target_cols}
        self.models = []

        for i, (ix_train, ix_cv) in enumerate(v_ix_cv):
            X_train, X_cv = X.iloc[ix_train].copy(), X.iloc[ix_cv].copy()
            y_train, y_cv = y.iloc[ix_train].copy(), y.iloc[ix_cv].copy()

            fold_paths = {}
            for target in self.target_cols:
                train_df = pd.concat([X_train.reset_index(drop=True), y_train[[target]].reset_index(drop=True)], axis=1)

                predictor_path = os.path.join(
                    Paths.path_models,
                    f'autogluon_{target}_{uuid4().hex}_fold_{i}'
                )
                if not os.path.exists(predictor_path):
                    # setup_outputdir(path=predictor_path)
                    predictor = (TabularPredictor(
                        label=target,
                        problem_type='regression',
                        eval_metric='root_mean_squared_error',
                        # path=predictor_path,
                    ).fit(
                        train_data=train_df,
                        presets=params['presets'],
                        time_limit=params['time_limit'],
                        hyperparameters=params['hyperparameters'],
                        num_bag_folds=params['num_bag_folds'],
                        num_bag_sets=params['num_bag_sets'],
                        num_stack_levels=params['num_stack_levels'],
                        verbosity=params['verbosity'],
                        num_gpus=1
                    ))
                    predictor.save()
                else:
                    predictor = TabularPredictor.load(predictor_path)

                fold_paths[target] = predictor_path
                self._predictor_cache[predictor_path] = predictor

                y_pred = predictor.predict(X_cv)
                rmse = float(np.sqrt(np.mean((y_cv[target].values - y_pred.values) ** 2)))
                scores_by_target[target].append(rmse)

            self.models.append(fold_paths)

        # final refit on all data
        self.final_model = {}
        full_train = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        for target in self.target_cols:
            predictor_path = os.path.join(
                Paths.path_models,
                f'autogluon_{target}_{uuid4().hex}_full'
            )
            if not os.path.exists(predictor_path):
                # setup_outputdir(path=predictor_path)
                predictor = TabularPredictor(
                    label=target,
                    problem_type='regression',
                    eval_metric='root_mean_squared_error',
                    # path=predictor_path,
                ).fit(
                    train_data=full_train[[*X.columns, target]],
                    presets=params['presets'],
                    time_limit=params['time_limit'],
                    hyperparameters=params['hyperparameters'],
                    num_bag_folds=params['num_bag_folds'],
                    num_bag_sets=params['num_bag_sets'],
                    num_stack_levels=params['num_stack_levels'],
                    verbosity=params['verbosity'],
                    num_gpus=1
                )
            else:
                predictor = TabularPredictor.load(predictor_path)

            self.final_model[target] = predictor_path
            self._predictor_cache[predictor_path] = predictor

        cv_mean_scores = {target: float(np.mean(vals)) for target, vals in scores_by_target.items()}
        info(f'CV RMSE theta: {cv_mean_scores["theta_y"]}')
        info(f'CV RMSE rho: {cv_mean_scores["rho_y"]}')
        info(f'CV RMSE psi: {cv_mean_scores["psi_y"]}')

        overall_mean = float(np.mean(list(cv_mean_scores.values())))
        info(f'CV rmse mean: {overall_mean}')

        if self.metric in ('theta', 'theta_y'):
            return 'RMSE', cv_mean_scores['theta_y']
        if self.metric in ('rho', 'rho_y'):
            return 'RMSE', cv_mean_scores['rho_y']
        if self.metric in ('psi', 'psi_y'):
            return 'RMSE', cv_mean_scores['psi_y']

        return 'RMSE', overall_mean

    def params_dflt(self) -> Dict:
        return {
            'presets': 'best_v150',
            'time_limit': 5*60,
            'hyperparameters': None,
            'num_bag_folds': 8,
            'num_bag_sets': 1,
            'num_stack_levels': 2,
            'verbosity': 2,
        }

    def _load_predictor(self, path: str) -> TabularPredictor:
        predictor = self._predictor_cache.get(path)
        if predictor is None:
            predictor = TabularPredictor.load(path)
            self._predictor_cache[path] = predictor
        return predictor

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), 'X must be a pandas DataFrame'

        if self.final_model:
            preds = {}
            for target, path in self.final_model.items():
                predictor = self._load_predictor(path)
                preds[target] = predictor.predict(X).values
            return pd.DataFrame(preds, index=X.index)

        if not self.models:
            raise ValueError('Estimator is not fitted')

        fold_preds = []
        for fold_paths in self.models:
            preds = {}
            for target, path in fold_paths.items():
                predictor = self._load_predictor(path)
                preds[target] = predictor.predict(X).values
            fold_preds.append(pd.DataFrame(preds, index=X.index))

        df_pred = sum(fold_preds) / len(fold_preds)
        return df_pred

    def predict(self, X):
        df_pred = self.predict_all(X)

        if self.metric in ('theta', 'theta_y'):
            return df_pred['theta_y'].values
        if self.metric in ('rho', 'rho_y'):
            return df_pred['rho_y'].values
        if self.metric in ('psi', 'psi_y'):
            return df_pred['psi_y'].values

        return df_pred

    def rmse(self, X, y, sample_weight=None):
        y_pred = self.predict(X)

        if isinstance(y, pd.Series):
            y_true = y.values
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        if isinstance(y, pd.DataFrame):
            pred_df = self.predict_all(X)
            errs = []
            for target in self.target_cols:
                errs.append(np.sqrt(np.mean((y[target].values - pred_df[target].values) ** 2)))
            return float(np.mean(errs))

        y_true = np.asarray(y)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def feature_names(self, model=None):
        return []

    @classmethod
    def model_name(cls, version: str, alpha: float = 0.5):
        return f'{cls.model_base_name}_{version}.joblib'
