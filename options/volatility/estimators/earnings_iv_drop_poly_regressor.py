import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import load

from shared.paths import Paths
from shared.constants import model_nm_earnings_iv_drop_regressor


class EarningsIVDropPolyRegressorV3(BaseEstimator, ClassifierMixin):
    def __init__(self, model_nm=None):
        self.pipeline = load(os.path.join(Paths.path_models, model_nm or model_nm_earnings_iv_drop_regressor))

    def predict(self, df: pd.DataFrame, min_moneyness=-99, max_moneyness=99, min_tenor=0) -> np.array:
        x = df.copy(True)
        x.loc[:, 'moneyness_fwd_ln'] = np.minimum(np.maximum(x.loc[:, 'moneyness_fwd_ln'], min_moneyness), max_moneyness)
        x.loc[:, 'tenor'] = np.maximum(x.loc[:, 'tenor'], min_tenor)
        return self.pipeline.predict(x)

    def load_model(self, path):
        self.pipeline = load(path)
        return self
