from .enum_utils import EnumStr
from enum import Enum


class Features(EnumStr, Enum):
    y_post_valley = 'y_post_valley'
    y_pre_valley = 'y_pre_valley'
    y_post_peak = 'y_post_peak'
    y_pre_peak = 'y_pre_peak'
    y_valley = 'y_valley'
    y_peak = 'y_peak'
    y_long_1 = 'y_long_1'
    y_short_1 = 'y_short_1'
    regr_lgb_close_30 = 'regr_lgb_close_30'
    regr_lgb_close_60 = 'regr_lgb_close_60'
    regr_lgb_close_120 = 'regr_lgb_close_120'
    regr_lgb_close_300 = 'regr_lgb_close_300'
    regr_lgb_close_600 = 'regr_lgb_close_600'
    regr_lgb_close_1800 = 'regr_lgb_close_1800'
    rl_risk_reward_ls_actual = 'rl_risk_reward_ls_actual'
    rl_risk_reward_neutral_actual = 'rl_risk_reward_neutral_actual'
