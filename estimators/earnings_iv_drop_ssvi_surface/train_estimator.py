import numpy as np
import pandas as pd
import plotly.graph_objs as go
import os

from dataclasses import dataclass, field
from datetime import datetime, time
from itertools import chain
from typing import Dict, List, Tuple, Callable
from joblib import dump
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold

from estimators.transformer import (
    EnrichReleaseDatesTransformer, AddBusinessDaysTransformer,
    PivotMetricsTransformer, AddTenorTransformer,
    FinalizeFrameTransformer, FeatureConfig, AddSpotFingerprintTransformer, AddEarningsJumpLagsTransformer, EnrichSpotTransformer, EnrichReturnTransformer, AddVegaTransformer
)
from options.frame_builder import available_sym_dates
from options.helper import cache_to_disk  # no explicity decorator
from options.surfaces.processors import get_ivs_mp, ScopePrePost  # jl
from estimators.earnings_iv_drop_ssvi_surface.estimator_lgbm import ESSVITreeEstimator
from options.types.equity import Equity  # jl
from options.types.iv_surface_essvi import IVSurface  # jl
from options.types.scope_pre_post import scoped_dates  # jl
from options.types.sym_date import SymDate
from shared.constants import dt_fmt_ymdhms  # jl
from shared.modules.logger import info
from shared.paths import Paths  # jl
from shared.plotting import show
from shared.utils.decorators import time_it

TARGET_COLS = ['theta_y', 'rho_y', 'psi_y']
WEIGHT_COL = 'vega'
STACKING_ORDER = ['theta', 'rho', 'psi']
frame_x_to_y_merge_keys = ['underlying', 'release_date', 'tenor_dt']

@dataclass
class PreSurfaceFeatureSource:
    sym_dates: List[SymDate]
    v_ivs: list
    df_points: pd.DataFrame
    df_pivot_base: pd.DataFrame


PreSurfaceFeatureFn = Callable[[pd.DataFrame, PreSurfaceFeatureSource, "TrainConfig"], pd.DataFrame]


def build_feature_columns_from_transformers(df: pd.DataFrame, feature_cfg: FeatureConfig) -> List[str]:
    """Derive feature columns from the transformers in the config."""
    feature_cols: List[str] = []

    for transformer in feature_cfg.transformers:
        transformer_cols = transformer.get_feature_columns()
        feature_cols.extend([c for c in transformer_cols if c in df.columns])

    feature_cols = sorted(set(feature_cols))
    return feature_cols



@dataclass
class DataConfig:
    mp: bool = False
    arb_free: bool = False
    min_surface_samples: int = 300
    post_event_min_time: time = time(12, 0)


@dataclass
class TrainConfig:
    n_cv_splits: int = 5
    alpha: float = 0.5
    model_version_prefix: str = 'f'
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    data_config: DataConfig = field(default_factory=DataConfig)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_cols: List[str]
    target_cols: List[str]
    meta_cols: List[str]
    weight_col: str = None


@dataclass
class TrainArtifacts:
    estimators: Dict[str, ESSVITreeEstimator]
    cv_scores: Dict[str, Tuple[str, float]]
    holdout_scores: Dict[str, float]
    train_bundle: DatasetBundle
    holdout_bundle: DatasetBundle | None


def split_sym_dates_by_symbol(
    tickers_train: List[str],
    tickers_holdout: List[str] | None = None,
    exclude: List[str] | None = None,
) -> Tuple[List[SymDate], List[SymDate]]:
    exclude = exclude or []
    tickers_holdout = tickers_holdout or []

    train_symbols = [t for t in tickers_train if t and t not in exclude and t not in tickers_holdout]
    holdout_symbols = [t for t in tickers_holdout if t not in exclude]

    sym_dates_train = available_sym_dates(','.join(train_symbols)) if train_symbols else []
    sym_dates_holdout = available_sym_dates(','.join(holdout_symbols)) if holdout_symbols else []

    sym_dates_train = [t for t in sym_dates_train if all(IVSurface.is_cached(Equity(t.symbol), [dt]) for dt in scoped_dates(t.date, ScopePrePost.mini_train))]
    sym_dates_holdout = [t for t in sym_dates_holdout if all(IVSurface.is_cached(Equity(t.symbol), [dt]) for dt in scoped_dates(t.date, ScopePrePost.mini_train))]

    info(f'Train symbols: {train_symbols}')
    info(f'Holdout symbols: {holdout_symbols}')
    info(f'# train sym_dates: {len(sym_dates_train)}')
    info(f'# holdout sym_dates: {len(sym_dates_holdout)}')

    return sym_dates_train, sym_dates_holdout


def build_cv_splits_by_symbol_release_date(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = list(map(tuple, df[['underlying', 'release_date']].drop_duplicates().values))
    info(f'CV grouping on {len(groups)} underlying/release_date combinations.')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    out: List[Tuple[np.ndarray, np.ndarray]] = []

    group_keys = df[['underlying', 'release_date']].apply(tuple, axis=1)
    for fold_ix, (ix_train_group, ix_cv_group) in enumerate(kf.split(groups)):
        train_groups = {groups[i] for i in ix_train_group}
        is_train = group_keys.isin(train_groups).values
        ix_train = df.index[is_train].values
        ix_cv = df.index[~is_train].values
        out.append((ix_train, ix_cv))
        info(f'CV split {fold_ix}: train={len(ix_train)} cv={len(ix_cv)}')

    return out


def _filter_surfaces(v_ivs, min_surface_samples: int):
    filtered = [ivs for ivs in v_ivs if sum(len(v.mny_fwd_ln) for v in ivs.calibration_items.values()) > min_surface_samples]
    info(f'Excluded {len(v_ivs) - len(filtered)} surfaces with too few samples. Remaining={len(filtered)}')
    return filtered


def _surface_datapoints_to_df(v_ivs) -> pd.DataFrame:
    return pd.DataFrame(chain(*[ivs.datapoints for ivs in v_ivs])).rename(columns={'model_param': 'metric'})


def build_pre_jump_observations(sym_dates: List[SymDate], cfg: TrainConfig, scope: str = ScopePrePost.mini_train_pipe) -> pd.DataFrame:
    v_ivs_x = get_ivs_mp(sym_dates, scope=scope, arb_free=cfg.data_config.arb_free, mp=cfg.data_config.mp)
    v_ivs_x = _filter_surfaces(v_ivs_x, cfg.data_config.min_surface_samples)

    for ivs in v_ivs_x:
        ivs.id = getattr(ivs, 'id', None) or __import__('uuid').uuid4()

    df_x = _surface_datapoints_to_df(v_ivs_x)

    for transformer in cfg.feature_config.transformers:
        df_x = transformer.transform(df_x, v_ivs=v_ivs_x, sym_dates=sym_dates)

    assert df_x[['underlying', 'release_date', 'tenor_dt', 'time_to_release_days', 'metric']].duplicated().sum() == 0

    return df_x

def build_post_target_frame(sym_dates: List[SymDate], cfg: TrainConfig) -> pd.DataFrame:
    v_ivs_y = get_ivs_mp(sym_dates, scope=ScopePrePost.mini_post, arb_free=cfg.data_config.arb_free, mp=cfg.data_config.mp)

    for ivs in v_ivs_y:
        ivs.id = getattr(ivs, 'id', None) or __import__('uuid').uuid4()

    df_y = _surface_datapoints_to_df(v_ivs_y)
    df_y = df_y[df_y['ts_end'].dt.time > cfg.data_config.post_event_min_time]
    df_y = EnrichReleaseDatesTransformer().transform(df_y)
    df_y = AddBusinessDaysTransformer().transform(df_y)

    assert df_y[['underlying', 'release_date', 'tenor_dt', 'time_to_release_days', 'metric']].duplicated().sum() == 0

    return df_y


@time_it
def build_to_estimate_dataset(sym_dates: List[SymDate], cfg: TrainConfig) -> DatasetBundle:
    df_x = build_pre_jump_observations(sym_dates, cfg, scope=ScopePrePost.live_trading)

    key = list(set(df_x.columns).difference({'metric', 'value'}))
    df_x_pivot = df_x.pivot(index=key, columns=['metric']).reset_index(drop=False)
    df_x_pivot.columns = [
        f"{b}" if a == "value" else a
        for a, b in df_x_pivot.columns
    ]

    # ── aggregate vega to tenor-level (mean across strikes/rights per tenor) ──
    if WEIGHT_COL in df_x.columns:
        vega_by_tenor = (
            df_x
            .groupby(frame_x_to_y_merge_keys)["vega"]
            .mean()
            .reset_index()
            .rename(columns={"vega": WEIGHT_COL})
        )
        df_x_pivot = df_x_pivot.merge(vega_by_tenor, on=frame_x_to_y_merge_keys, how="left")
    # vega end

    df = df_x_pivot.reset_index(drop=True)

    metrics_col = ['theta', 'psi', 'rho']
    ix_drop = set(df.index[df[metrics_col].isna().sum(axis=1) > 0])
    info(f'Rows before NA drop: {len(df)} | # Missing metrics: {len(ix_drop)}')
    df = df.drop(ix_drop).reset_index(drop=True)
    info(f'Rows after NA drop: {len(df)}')

    feature_cols = build_feature_columns_from_transformers(df, cfg.feature_config) + ['theta', 'psi', 'rho']

    info(f'Using features: {feature_cols}')
    return DatasetBundle(
        df=df,
        feature_cols=feature_cols,
        target_cols=TARGET_COLS,
        meta_cols=frame_x_to_y_merge_keys,
    )


@time_it
def build_training_dataset(sym_dates: List[SymDate], cfg: TrainConfig) -> DatasetBundle:
    df_x = build_pre_jump_observations(sym_dates, cfg)
    df_y = build_post_target_frame(sym_dates, cfg)

    df_y_pivot = (df_y[frame_x_to_y_merge_keys + ['metric', 'value']]
                  .pivot(index=frame_x_to_y_merge_keys, columns=['metric'])
                  .reset_index()
                  )
    df_y_pivot.columns = [
        f"{b}_y" if a == "value" else a
        for a, b in df_y_pivot.columns
    ]

    key = list(set(df_x.columns).difference({'metric', 'value'}))
    df_x_pivot = df_x.pivot(index=key, columns=['metric']).reset_index(drop=False)
    df_x_pivot.columns = [
        f"{b}" if a == "value" else a
        for a, b in df_x_pivot.columns
    ]

    # ── aggregate vega to tenor-level (mean across strikes/rights per tenor) ──
    if WEIGHT_COL in df_x.columns:
        vega_by_tenor = (
            df_x
            .groupby(frame_x_to_y_merge_keys)["vega"]
            .mean()
            .reset_index()
            .rename(columns={"vega": WEIGHT_COL})
        )
        df_x_pivot = df_x_pivot.merge(vega_by_tenor, on=frame_x_to_y_merge_keys, how="left")
    # vega end

    df = df_x_pivot.merge(
        df_y_pivot,
        on=frame_x_to_y_merge_keys,
        how='left',
        suffixes=('', '_y'),
        # validate='one_to_many',
    ).reset_index(drop=True)

    metrics_col = ['theta', 'psi', 'rho']
    metrics_col += [f'{c}_y' for c in metrics_col]
    ix_drop = set(df.index[df[metrics_col].isna().sum(axis=1) > 0])
    info(f'Rows before NA drop: {len(df)} | # Missing metrics: {len(ix_drop)}')
    df = df.drop(ix_drop).reset_index(drop=True)
    info(f'Rows after NA drop: {len(df)}')

    feature_cols = build_feature_columns_from_transformers(df, cfg.feature_config) + ['theta', 'psi', 'rho']

    info(f'Using features: {feature_cols}')
    return DatasetBundle(
        df=df,
        feature_cols=feature_cols,
        target_cols=TARGET_COLS,
        meta_cols=frame_x_to_y_merge_keys,
        # weight_col=WEIGHT_COL
    )


def get_xy(bundle: DatasetBundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = bundle.df[bundle.feature_cols].reset_index(drop=True)
    y = bundle.df[bundle.target_cols].reset_index(drop=True)
    return x, y


def predict_with_sequential_stacking(
    estimators: Dict[str, ESSVITreeEstimator],
    x: pd.DataFrame,
    stacking_order: List[str] = None,
) -> pd.DataFrame:
    stacking_order = stacking_order or STACKING_ORDER
    x_work = x.copy()
    preds = {}

    for metric in stacking_order:
        metric_y = f'{metric}_y'
        pred = estimators[metric].predict(x_work)
        preds[metric_y] = pred
        x_work[metric_y] = pred

    return pd.DataFrame(preds, index=x.index)


def plot_feature_importance(estimators: Dict[str, ESSVITreeEstimator], x_train: pd.DataFrame, top_n: int = 20, fn: str = 'ssvi_feature_importance.html'):
    fig = make_subplots(rows=3, cols=1, subplot_titles=[f'Feature importance: {m}' for m in STACKING_ORDER])

    for row_ix, metric in enumerate(STACKING_ORDER, start=1):
        model = estimators[metric].models[0]
        names = model.feature_name()
        values = model.feature_importance(importance_type='gain')

        df_imp = pd.DataFrame({'feature': names, 'importance': values}).sort_values('importance', ascending=False).head(top_n)
        df_imp = df_imp.iloc[::-1]

        fig.add_trace(
            go.Bar(
                x=df_imp['importance'],
                y=df_imp['feature'],
                orientation='h',
                name=metric,
            ),
            row=row_ix,
            col=1,
        )

    fig.update_layout(height=1000, title='SSVI post-earnings feature importance')
    show(fig, fn=fn, open_browser=True)


def plot_predictions_vs_actual(y_true: pd.DataFrame, y_pred: pd.DataFrame, fn: str = 'ssvi_predictions_vs_actual.html'):
    fig = make_subplots(rows=1, cols=3, subplot_titles=['theta', 'rho', 'psi'])

    for col_ix, metric in enumerate(STACKING_ORDER, start=1):
        metric_y = f'{metric}_y'
        fig.add_trace(
            go.Scatter(
                x=y_true[metric_y],
                y=y_pred[metric_y],
                mode='markers',
                name=metric,
            ),
            row=1,
            col=col_ix,
        )

        low = min(float(y_true[metric_y].min()), float(y_pred[metric_y].min()))
        high = max(float(y_true[metric_y].max()), float(y_pred[metric_y].max()))
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode='lines',
                name=f'{metric} perfect',
            ),
            row=1,
            col=col_ix,
        )

    fig.update_layout(height=450, title='Predicted vs actual SSVI params')
    show(fig, fn=fn, open_browser=True)


def score_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame, tag: str) -> Dict[str, float]:
    scores = {}
    for metric in STACKING_ORDER:
        metric_y = f'{metric}_y'
        rmse = float(np.sqrt(np.mean((y_true[metric_y].values - y_pred[metric_y].values) ** 2)))
        scores[metric] = rmse
        info(f'{tag} RMSE {metric}: {rmse}')
    scores['mean'] = float(np.mean(list(scores.values())))
    info(f'{tag} RMSE mean: {scores["mean"]}')
    return scores


@cache_to_disk('prep_x_w_return', Paths.path_analysis_frames)
def prep_x_w_return(
    sym_dates: List[SymDate],
    ret:float,
    estimators: Dict[str, ESSVITreeEstimator] | None = None,
    use_oof_stacking: bool = True,
    alpha: float | None = None,
) -> pd.DataFrame:

    feature_config = FeatureConfig(
        transformers=[
            EnrichReleaseDatesTransformer(),
            AddBusinessDaysTransformer(),
            AddTenorTransformer(),
            AddEarningsJumpLagsTransformer(),
            AddSpotFingerprintTransformer(lookback_days=(20,)),
            EnrichSpotTransformer(),
            EnrichReturnTransformer(ret=ret),
        ]
    )
    cfg = TrainConfig(
        n_cv_splits=3,
        alpha=0.5 if alpha is None else alpha,
        feature_config=feature_config,
        data_config=DataConfig(
            mp=False,
            arb_free=False,
            min_surface_samples=500,
            post_event_min_time=time(12, 0),
        ),
    )
    bundle = build_to_estimate_dataset(sym_dates, cfg)
    df = bundle.df

    if use_oof_stacking:
        if estimators is None:
            raise ValueError("estimators must be provided when use_oof_stacking=True")

        x_stack = bundle.df[bundle.feature_cols].reset_index(drop=True)

        for metric in STACKING_ORDER:
            metric_y = f"{metric}_y"
            stack_col = f"{metric_y}_stack"
            x_stack[stack_col] = estimators[metric].predict(x_stack)
            df[stack_col] = x_stack[stack_col].values

    return df


def run_pipeline(
    tickers_train: List[str],
    tickers_holdout: List[str] | None = None,
    cfg: TrainConfig | None = None,
) -> TrainArtifacts:
    cfg = cfg or TrainConfig()

    sym_dates_train, sym_dates_holdout = split_sym_dates_by_symbol(
        tickers_train=tickers_train,
        tickers_holdout=tickers_holdout or [],
    )

    train_bundle = build_training_dataset(sym_dates_train, cfg)
    holdout_bundle = build_training_dataset(sym_dates_holdout, cfg) if sym_dates_holdout else None

    return train_tree_regressor(
        train_bundle=train_bundle,
        holdout_bundle=holdout_bundle,
        cfg=cfg,
    )


def fit_estimator_oof(
    x: pd.DataFrame,
    y: pd.Series,
    v_ix_cv: List[Tuple[np.ndarray, np.ndarray]],
    metric_y: str,
    alpha: float | None = None,
    weight: pd.Series | None = None,
) -> Tuple[ESSVITreeEstimator, np.ndarray, Tuple[str, float]]:
    est = ESSVITreeEstimator(metric_y)
    stat, score = est.fit(x, y, v_ix_cv,
                          weight=weight.reset_index(drop=True) if weight is not None else None,
                          alpha=alpha)

    oof_pred = np.full(len(x), np.nan, dtype=float)
    for ix_train, ix_cv in v_ix_cv:
        fold_est = ESSVITreeEstimator(metric_y)
        fold_est.fit(
            x.iloc[ix_train].reset_index(drop=True),
            y.iloc[ix_train].reset_index(drop=True),
            v_ix_cv=[(
                np.arange(len(ix_train)),
                np.arange(len(ix_train)),
            )],
            weight=weight.iloc[ix_train].reset_index(drop=True) if weight is not None else None,  # <-- add
            alpha=alpha,
        )
        oof_pred[ix_cv] = fold_est.predict(x.iloc[ix_cv])

    if np.isnan(oof_pred).any():
        raise ValueError(f'OOF predictions contain NaN for {metric_y}')

    return est, oof_pred, (stat, score)


def predict_with_oof_stacking(
    estimators: Dict[str, ESSVITreeEstimator],
    x: pd.DataFrame,
    stacking_order: List[str] = None,
    stack_feature_suffix: str = '_stack',
) -> pd.DataFrame:
    stacking_order = stacking_order or STACKING_ORDER
    x_work = x.copy()
    preds = {}

    for metric in stacking_order:
        metric_y = f'{metric}_y'
        pred = estimators[metric].predict(x_work)
        preds[metric_y] = pred
        x_work[f'{metric_y}{stack_feature_suffix}'] = pred

    return pd.DataFrame(preds, index=x.index)


@time_it
def train_tree_regressor(
    train_bundle: DatasetBundle,
    holdout_bundle: DatasetBundle | None = None,
    cfg: TrainConfig | None = None,
) -> TrainArtifacts:
    cfg = cfg or TrainConfig()

    x_train, y_train = get_xy(train_bundle)
    v_ix_cv = build_cv_splits_by_symbol_release_date(train_bundle.df, n_splits=cfg.n_cv_splits)

    # ── resolve sample weights ─────────────────────────────────────────────
    w_train: pd.Series | None = None
    if train_bundle.weight_col and train_bundle.weight_col in train_bundle.df.columns:
        w_train = train_bundle.df[train_bundle.weight_col].reset_index(drop=True)
        info(f'Using sample weights from column: {train_bundle.weight_col}')

    estimators: Dict[str, ESSVITreeEstimator] = {}
    cv_scores: Dict[str, Tuple[str, float]] = {}

    x_train_stack = x_train.copy()
    x_holdout_stack = None
    y_ho = None
    if holdout_bundle is not None:
        x_holdout_stack, y_ho = get_xy(holdout_bundle)
        x_holdout_stack = x_holdout_stack.copy()

    for metric in STACKING_ORDER:
        metric_y = f'{metric}_y'
        est, oof_pred, cv_stat = fit_estimator_oof(
            x=x_train_stack,
            y=y_train[metric_y],
            v_ix_cv=v_ix_cv,
            metric_y=metric_y,
            alpha=cfg.alpha,
            weight=w_train,
        )
        estimators[metric] = est
        cv_scores[metric] = cv_stat

        x_train_stack[f'{metric_y}_stack'] = oof_pred

        if x_holdout_stack is not None:
            x_holdout_stack[f'{metric_y}_stack'] = est.predict(x_holdout_stack)

        info(f'Finished OOF stacked target: {metric}')

    version = f'{cfg.model_version_prefix}_{datetime.now().strftime(dt_fmt_ymdhms)}'
    dump(
        estimators,
        os.path.join(Paths.path_models, ESSVITreeEstimator.model_name(version, cfg.alpha)),
    )

    for metric, (stat, score) in cv_scores.items():
        info(f'CV {stat} {metric}: {score}')

    holdout_scores = {}
    if holdout_bundle is not None:
        pred_ho = pd.DataFrame(
            {f'{metric}_y': estimators[metric].predict(x_holdout_stack) for metric in STACKING_ORDER},
            index=x_holdout_stack.index,
        )
        holdout_scores = score_predictions(y_ho, pred_ho, tag='HO')
        plot_predictions_vs_actual(y_ho, pred_ho)

    plot_feature_importance(estimators, x_train_stack)

    return TrainArtifacts(
        estimators=estimators,
        cv_scores=cv_scores,
        holdout_scores=holdout_scores,
        train_bundle=train_bundle,
        holdout_bundle=holdout_bundle,
    )

if __name__ == "__main__":
    """
    - how can open equity volume and contract eq VOLUME and option open INTEREST help?
    - distribution parameters of open interest around strike could be good? could bucket open interest somehow, a surface. 9 bucket / low/mid/high on strike/maturity ?!
    """
    tickers_holdout = ['DAL', 'HPE', 'PEP']
    exclude = {'LNG'}
    # tickers = 'FDX,NKE'#,TGT,PEP,ABT,HPE'
    tickers_all = [s.upper() for s in os.listdir(Paths.path_data.joinpath('option', 'usa', 'second'))]
    tickers_train = list(set(tickers_all) - set(tickers_holdout) - exclude)

    feature_config = FeatureConfig(
        transformers=[
            EnrichReleaseDatesTransformer(),
            AddBusinessDaysTransformer(),
            # PivotMetricsTransformer(),
            # AddIvJumpsTransformer(),
            AddTenorTransformer(),
            AddEarningsJumpLagsTransformer(),
            AddSpotFingerprintTransformer(lookback_days=(20,)),
            EnrichSpotTransformer(),
            EnrichReturnTransformer(),
            # AddVegaTransformer(),  # missing strike or ATM IVs
            # AddLaggedFeaturesTransformer(max_lag_days=20),
            # AddRelativeDifferenceLagsTransformer(max_lag_days=20),
            # FinalizeFrameTransformer(),
        ]
    )

    for alpha in (0.1, 0.5, 0.9):

        cfg = TrainConfig(
            n_cv_splits=3,
            alpha=alpha,
            feature_config=feature_config,
            data_config=DataConfig(
                mp=False,
                arb_free=False,
                min_surface_samples=500,
                post_event_min_time=time(12, 0),
            ),
        )

        run_pipeline(
            tickers_train=tickers_train,
            tickers_holdout=tickers_holdout,
            cfg=cfg,
        )
        # 2026-05-16 16:24:54,940 - INFO CV quantile mean: 0.007149099996798554
        # 2026-05-16 16:24:54,947 - INFO Finished OOF stacked target: psi
        # 2026-05-16 16:24:55,132 - INFO CV QUANTILE theta: 0.0015722902319755666
        # 2026-05-16 16:24:55,132 - INFO CV QUANTILE rho: 0.05448534904147509
        # 2026-05-16 16:24:55,132 - INFO CV QUANTILE psi: 0.012286998282127962
        # 2026-05-16 16:24:55,140 - INFO HO RMSE theta: 0.016049136650138497
        # 2026-05-16 16:24:55,140 - INFO HO RMSE rho: 0.4027244733656254
        # 2026-05-16 16:24:55,140 - INFO HO RMSE psi: 0.08056444153025995
        # 2026-05-16 16:24:55,140 - INFO HO RMSE mean: 0.16644601718200794
        # 2026-05-16 16:24:55,409 - INFO Total Runtime train_tree_regressor: 2.7 seconds

        # After deleting bad ones...
        # 2026-05-16 23:05:56,593 - INFO CV quantile mean: 0.004539554933906735
        # 2026-05-16 23:05:56,597 - INFO Finished OOF stacked target: psi
        # 2026-05-16 23:05:56,643 - INFO CV QUANTILE theta: 0.001550104270840049
        # 2026-05-16 23:05:56,643 - INFO CV QUANTILE rho: 0.04661429641395016
        # 2026-05-16 23:05:56,643 - INFO CV QUANTILE psi: 0.010513233535981215
        # 2026-05-16 23:05:56,792 - INFO Total Runtime train_tree_regressor: 2.9 seconds