import numpy as np
import pickle
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from estimators.utils import ivs_from_maxbid_minask_iv_enriched, ps_in_slices, plot_surface, plot_scatter_3d
from options.types.equity import Equity
from shared.modules.logger import info

xy_store_fn = 'earnings_iv_drop.pkl'
moneyness_fit = 'moneyness_fwd_ln'
iv_col_nm = 'mid_iv'
# iv_col_nm = 'mid_price_iv'
min_mny = 0.8
max_mny = 1.2


def snap_post_release_ivs(df_in: pd.DataFrame, equity: Equity) -> pd.DataFrame:
    df = ivs_from_maxbid_minask_iv_enriched(df_in, equity)
    return df


def compare_polynomials(xy, col_fit, sample_weight_col):
    degrees = range(1, 10)

    scores = []
    for degree in degrees:
        model = fit_polynomial(xy, col_fit, sample_weight_col=sample_weight_col, degree=degree)
        scores.append(score_model(model, xy, col_fit, sample_weight_col))

    df_score = pd.concat(scores, axis=1).T
    df_score.insert(0, 'degree', list(degrees))
    # respective coefficients/intercepts for different degrees
    pd.set_option('display.max_colwidth', 180)
    print(df_score[['degree', 'r2_score', 'MAE', 'MSE']])
    return df_score


def fit_polynomial(xy_in, col_fit: str, sample_weight_col='vega_mid_iv', degree=7) -> Pipeline:
    xy = xy_in[(~xy_in[col_fit].isna()) & (xy_in[col_fit] != np.inf)]
    if col_fit in ('d_mid_iv_ratio', 'd_mid_iv_ratio_sub_hv_normed'):
        X = xy[[moneyness_fit, 'tenor']]
    else:
        X = xy[['tenor']]

    y = np.array(xy[col_fit].astype(float)).reshape(-1, 1)
    if sample_weight_col:
        sample_weight = xy[sample_weight_col].values
    else:
        sample_weight = None

    print(f'Fitting model on {len(y)} samples with a polynomial of degree {degree}.')
    model = Pipeline([('poly_features', PolynomialFeatures(degree=degree)),
                         ('model', LinearRegression())])
    model.fit(X, y, model__sample_weight=sample_weight)

    return model


def score_model(model: Pipeline, xy, col_fit, sample_weight_col='vega_mid_iv') -> pd.Series:
    if col_fit == 'd_mid_iv_ratio':
        d_iv_ratio = model.predict(xy[[moneyness_fit, 'tenor']]).flatten()
        y_iv_pred = xy[iv_col_nm] * (1 + d_iv_ratio)

    if col_fit == 'd_mid_iv_ratio_sub_hv_normed':
        d_iv_ratio = model.predict(xy[[moneyness_fit, 'tenor']]).flatten()
        y_iv_pred = xy['mid_iv_m_hv'] * (1 + d_iv_ratio) + xy['hv_mean']

    elif col_fit == 'd_jump_iv_ratio':
        d_iv_ratio = model.predict(xy[['tenor']]).flatten()
        y_iv_pred = (xy[iv_col_nm] + xy['jump_iv'] * d_iv_ratio).values

    y_iv = xy[f'mid_iv_1'].values
    sample_weight = xy[sample_weight_col].values if sample_weight_col else None
    xy['error'] = y_iv - y_iv_pred
    xy['error_weighted'] = xy['error'].values * sample_weight

    r2_score = metrics.r2_score(y_iv, y_iv_pred, sample_weight=sample_weight)
    mae = mean_absolute_error(y_iv, y_iv_pred, sample_weight=sample_weight)
    mse = mean_squared_error(y_iv, y_iv_pred, squared=False, sample_weight=sample_weight)
    coeff = model.steps[1][1].coef_
    icpt = model.steps[1][1].intercept_

    score = pd.Series([r2_score, mae, mse, len(xy), coeff, icpt], index=['r2_score', 'MAE', 'MSE', 'samples', 'Coefficients', 'Intercepts', ])

    return score


def train_data_preprocessing(xy: pd.DataFrame, col_fit: str, plot=False) -> pd.DataFrame:
    # Enriching Historical Vola
    # for ix, s_df in xy.groupby(['symbol', 'release_date']):
    #     dt, res = ix
    #     xy.loc[s_df.index, 'hv_mean'] = get_hv_base(s_df['equity'].iloc[0], dt, res)

    if col_fit == 'd_mid_iv_ratio':
        if col_fit not in xy.columns:
            xy[f'd_mid_iv'] = xy[f'mid_iv_1'] - xy[iv_col_nm]
            xy[col_fit] = xy[f'd_mid_iv'] / xy[iv_col_nm]
            # print(xy[col_fit].sum())

        xy = xy[xy[col_fit] < 5]

    elif col_fit == 'd_mid_iv_ratio_sub_hv_normed':
        if col_fit not in xy.columns:
            xy[f'mid_iv_1_m_hv'] = xy[f'mid_iv_1'] - xy['hv_mean']
            xy['mid_iv_m_hv'] = xy[iv_col_nm] - xy['hv_mean']
            xy[f'd_mid_iv_m_hv'] = xy[f'mid_iv_1_m_hv'] - xy['mid_iv_m_hv']
            xy[col_fit] = xy[f'd_mid_iv_m_hv'] / xy['mid_iv_m_hv']
            # print(xy[col_fit].sum())

        xy = xy[xy[col_fit] < 5]

    elif col_fit == 'd_jump_iv_ratio':
        xy[f'd_mid_iv'] = xy[f'mid_iv_1'] - xy[iv_col_nm]
        xy[f'd_jump_iv_ratio'] = xy[f'd_mid_iv'] / xy['jump_iv']

        if plot:
            xy_tmp = xy[abs(xy['d_jump_iv_ratio']) < 10]
            plot_scatter_3d(xy_tmp[moneyness_fit], xy['tenor'], xy_tmp['d_jump_iv_ratio'])
            plot_scatter_3d(xy_tmp[moneyness_fit], xy['tenor'], xy_tmp['d_jump_iv_ratio'])

        xy = xy[(xy['jump_iv'] < 4) & (xy['jump_iv'] > 0)]
        xy = xy[abs(xy['d_jump_iv_ratio']) < 10]

    xy = xy[(~xy[col_fit].isna()) & (xy[col_fit] != np.inf)]

    x = xy[col_fit]
    mi, ma = [x.mean() - 3 * x.std(), x.mean() + 3 * x.std()]
    xy = xy[(xy[col_fit] > mi) & (xy[col_fit] < ma)]

    if 'moneyness_fwd_ln' in xy.columns:
        xy = xy[(xy['moneyness_fwd_ln'] > min_mny-1) & (xy['moneyness_fwd_ln'] < max_mny-1)]
    elif 'moneyness' in xy.columns:
        xy = xy[(xy['moneyness'] > min_mny) & (xy['moneyness'] < max_mny)]

    xy = xy[(xy['dte_front'] < 7)]  # Only if release is imminent

    return xy


def train(xy: pd.DataFrame, col_fit, col_weight='vega_mid_iv'):
    # df_score = compare_polynomials(xy, col_fit=col_fit, sample_weight_col=col_weight)

    model = fit_polynomial(xy, col_fit, sample_weight_col=col_weight, degree=7)
    score_model(model, xy, col_fit, sample_weight_col=col_weight)

    return model


def adjust_weight_by_sample_size_per_symbol_release(xy, col_weight: str, dry_run=True):
    # Sample weighting
    max_count = 0
    for ix, s_df in xy.groupby(['symbol', 'release_date']):
        max_count = max(max_count, len(s_df))
    info(f'Sample weighting. Increase importance of release with few samples only. Max count: {max_count}')
    for ix, s_df in xy.groupby(['symbol', 'release_date']):
        info(f'dry_run={dry_run}, Weighting {ix}. # samples: {len(s_df)}. Multiply vega by {max_count / len(s_df)}')
        if not dry_run:
            xy.loc[s_df.index, col_weight] *= max_count / len(s_df)
    return xy


def run(plot=True):
    """
    weight. vega - consider using volume as a weight
    refactor using 5-fold cv for evaluation. final training on all data.

    Think about how to refactor
    """
    # col_fit = 'd_jump_iv_ratio'
    col_fit = 'd_mid_iv_ratio'
    # col_fit = 'd_mid_iv_ratio_sub_hv_normed'

    weight = 'vega_mid_iv'
    axes = ('moneyness_fwd_ln', 'tenor')

    with open(xy_store_fn, 'rb') as f:
        xy = pickle.load(f)

    # Further preprocessing
    xy = train_data_preprocessing(xy, col_fit=col_fit)
    # xy = adjust_weight_by_sample_size_per_symbol_release(xy, col_weight=weight)

    if plot:
        plot_surface(xy, weight, axes)
        # plot_surface(xy[xy['tenor'] > 1], 'd_mid_iv', axes=('implied_move', 'tenor'))
        # plot_surface(xy[xy['tenor'] > 1], 'd_mid_iv_ratio', axes=('implied_move', 'tenor'))
        # xy.groupby(['symbol', 'release_date']).agg({'implied_move': 'mean'}).sort_values('implied_move')
        # plot_surface(xy, 'mid_iv_1', axes)
        # plot_surface(xy, 'd_mid_iv', axes)

    # Score jump IV estimation
    # xy['neg_jump_iv'] = -xy['jump_iv']
    # plot_surface(xy, 'neg_jump_iv')
    # print(metrics.r2_score(xy['mid_iv_1'].values, xy[iv_col_nm] + xy['neg_jump_iv'].values, sample_weight=xy[weight].values))
    # print(mean_absolute_error(xy['mid_iv_1'].values, xy[iv_col_nm] + xy['neg_jump_iv'].values, sample_weight=xy[weight].values))

    xy_slices = np.array(sorted(set(xy[['symbol', 'release_date']].to_records(index=False).tolist())))
    if len(xy_slices) > 1:
        info(f'Fitting model on {len(xy)} samples with {len(xy_slices)} slices.')
        kf = KFold(n_splits=min(5, len(xy_slices)))
        info(kf)
        scores = []
        for i, (train_index, test_index) in enumerate(kf.split(xy_slices)):
            train_slices = xy_slices[train_index]
            info(f"Fold {i}:")
            # print(f"  Train: index={train_index}, slices: {train_slices}")
            # print(f"  Test:  index={test_index}, slices: {xy_slices[test_index]}")

            ix_train = xy.apply(lambda ps: ps_in_slices(ps, train_slices), axis=1)
            model = train(xy[ix_train], col_fit, weight)
            scores.append(score_model(model, xy[~ix_train], col_fit, sample_weight_col=weight))
    else:
        model = train(xy, col_fit, weight)
        scores = [score_model(model, xy, col_fit, sample_weight_col=weight)]
    info(pd.concat(scores, axis=1).T[['r2_score', 'MAE', 'MSE']])
    info(pd.concat(scores, axis=1).T[['r2_score', 'MAE', 'MSE']].mean(axis=0))

    # Train on all data
    model = train(xy, col_fit, weight)
    print(score_model(model, xy, col_fit, sample_weight_col=weight)[['r2_score', 'MAE', 'MSE']])

    if plot:
        # plot_surface(xy, 'd_mid_iv', axes)
        plot_surface(xy, 'error', axes)
        plot_surface(xy, 'error_weighted', axes)

    # dump(model, os.path.join(Paths.path_models, f'earnings_iv_drop_regressor_v3_{date.today()}.joblib'))

    # score_by_dte(xy, model, col_fit, weight)
    # score_by_implied_move_bucket(xy, model, col_fit, weight)


def score_by_atm_iv_bucket(xy, model, col_fit, weight):

    for bin in pd.cut(xy['atm_iv'], 10).unique():
        samples = xy[xy['atm_iv'].between(bin.left, bin.right)]
        print('ATM IV:', bin, f'# Samples: {len(samples)}')
        if len(samples) > 0:
            print(score_model(model, samples, col_fit, sample_weight_col=weight)[['r2_score', 'MAE', 'MSE']])


def score_by_hv_bucket(xy, model, col_fit, weight):
    for bin in pd.cut(xy['hv_mean'], 5).unique():
        samples = xy[xy['hv_mean'].between(bin.left, bin.right)]
        print('HV:', bin, f'# Samples: {len(samples)}')
        if len(samples) > 0:
            print(score_model(model, samples, col_fit, sample_weight_col=weight)[['r2_score', 'MAE', 'MSE']])


def score_by_implied_move_bucket(xy, model, col_fit, weight):
    for bin in pd.cut(xy['implied_move'].dropna().values, 5).unique():
        samples = xy[xy['implied_move'].between(bin.left, bin.right)]
        print('ImpliedMove:', bin, f'# Samples: {len(samples)}')
        if len(samples) > 0:
            print(score_model(model, samples, col_fit, sample_weight_col=weight)[['r2_score', 'MAE', 'MSE']])


def score_by_dte(xy, model, col_fit, weight):
    # Score by DTE
    for dte, s_df in xy.groupby('dte_front'):
        print('DTE:', dte)
        print(score_model(model, s_df, col_fit, sample_weight_col=weight)[['r2_score', 'MAE', 'MSE']])


if __name__ == "__main__":
    """
    1. Compared to previous model, calculate the pct drop off the estimated jump IV.
        Use data from 1 month ago as most simple example.
        If promising, load more data to to estimated front month diffusion IV.
    2. Test using moneyness_fwd and moneyness_fwd_ln.
    3. Then may wanna focus on market IV, volume, open interest and another modeling strategy.
    
    2024-08-21 10:40:34,062 - INFO    
    r2_score       MAE       MSE
    0  0.823316  0.021461  0.039461
    1  0.780822  0.026755  0.052678
    2  0.790693  0.024521   0.06226
    3  0.819683  0.032242  0.060463
    4  0.756553  0.023329  0.040702
    
    2024-08-21 10:40:34,068 - INFO 
    r2_score    0.794213
    MAE         0.025662
    MSE         0.051113
    
    Fitting model on 35535 samples with a polynomial of degree 7.
    r2_score    0.836094
    MAE         0.024387
    MSE         0.049947
    """
    run()
