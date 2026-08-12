import os
import pickle

from sklearn import metrics
from shared.paths import Paths


def evaluate_model(m: object, xy, f_predict, nm):
    y = xy[f'mid_iv_1'].values
    x = f_predict(m, xy)
    xy['error'] = (x - y)  # * xy['vega_mid_iv'].values

    print(f"# Samples: {len(y)}")
    print(f"R2            : {metrics.r2_score(y, x)}")
    print(f"R2  (weighted): {metrics.r2_score(y, x, sample_weight=xy['vega_mid_iv'].values)}")
    print(f"MAE           : {metrics.mean_absolute_error(y, x)}")
    print(f"MAE (weighted): {metrics.mean_absolute_error(y, x, sample_weight=xy['vega_mid_iv'].values)}")
    print(f"RMSE          : {metrics.mean_squared_error(y, x, squared=False)}")
    print(f"RMSE(weighted): {metrics.mean_squared_error(y, x, sample_weight=xy['vega_mid_iv'].values, squared=False)}")

    # Error surface
    plot_surface(xy, 'error', axes=('moneyness', 'tenor'), fn=f'{nm}.html', scatter=True)


if __name__ == '__main__':
    """
    Load a fairly unprocessed dataset of option prices post-earnings after 12pm and calc the MAE/RMSE, % of Spread as error metrics for each right.
    """

    from options.volatility.estimators.earnings_iv_drop_poly_regressor import EarningsIVDropPolyRegressorV3 as Regressor1
    path_model = os.path.join(Paths.path_models, 'earnings_iv_drop_regressor_2024-04-29.joblib')
    path_modelb = os.path.join(Paths.path_models, 'earnings_iv_drop_regressor_old_2024-06-20.joblib')
    m1 = Regressor1().load_model(path_model)
    m1b = Regressor1().load_model(path_modelb)

    def f_predict1(m, xy):
        d_iv_ratio = m.predict(xy[['moneyness_fwd_ln', 'tenor']]).flatten()
        return xy['mid_iv'].values * (1 + d_iv_ratio)

    from estimators.earnings_iv_drop_polynomial.train_earnings_iv_drop_regressor_polynomial import EarningsIVDropPolyRegressor as Regressor2, plot_surface

    path_model = os.path.join(Paths.path_models, 'earnings_iv_drop_regressor2_2024-06-19.joblib')
    path_modelb = os.path.join(Paths.path_models, 'earnings_iv_drop_regressor2b_2024-06-20.joblib')
    m2 = Regressor2().load_model(path_model)
    m2b = Regressor2().load_model(path_modelb)

    def f_predict2(m, xy):
        ratio = m.predict(xy[['tenor']]).flatten()
        return xy['mid_iv'].values + xy['jump_iv'] * ratio

    xy_store_fn = 'earnings_iv_drop2.pkl'
    with open(xy_store_fn, 'rb') as f:
        xy = pickle.load(f)

    evaluate_model(m1, xy, f_predict1, 'error_1')
    evaluate_model(m1b, xy, f_predict1, 'error_1b')
    print('-' * 20)
    evaluate_model(m2, xy, f_predict2, 'error_2')
    evaluate_model(m2b, xy, f_predict2, 'error_2b')
