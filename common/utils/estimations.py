import numpy as np
import pandas as pd


def compute_quantile_loss(y_true, y_pred, quantile):
    """
    Parameters
    ----------
    y_true : 1d ndarray
        Target value.

    y_pred : 1d ndarray
        Predicted value.

    quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
    """
    residual = y_true - y_pred
    return np.maximum(quantile * residual, (quantile - 1) * residual)


def f1_score_weighted_returns(ix_pos_long: np.ndarray, ix_pos_short: np.ndarray, ix_neg: np.ndarray, ps_returns: pd.Series, ps_label: pd.Series):
    """
    x > 0 True, < 0 False. Weight: abs(x)
    Adjustment here. False positives here can actually make money... Exclude those, dont punish
    """
    pos_short = ps_returns.iloc[ix_pos_short] - ps_label.iloc[ix_pos_short]
    pos_long = ps_label.iloc[ix_pos_long] - ps_returns.iloc[ix_pos_long]
    pos = np.array(pos_short.tolist() + pos_long.tolist())
    neg = ps_label.iloc[ix_neg]

    tp = (1 + pos[np.where(pos > 0)[0]]).sum()
    # Exclude profitable returns, only include actuall lossy negative returns.
    fp_long = (2 - ps_label.iloc[np.where(ps_label.iloc[ix_pos_long] < 1)]).sum()
    fp_short = ps_label.iloc[np.where(ps_label.iloc[ix_pos_short] > 1)].sum()
    fp = fp_long + fp_short

    fn = neg.sum()

    f1 = tp / (tp + 0.5 * (fp + fn))
    print(f'Weighted F1 Score: {f1}')