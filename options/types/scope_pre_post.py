from datetime import date
from options.helper import add_trade_days


class ScopePrePost:
    pre: str = 'pre'
    post: str = 'post'
    mini_post: str = 'mini_post'
    all: str = 'all'
    train: str = 'train'
    mini_train: str = 'mini_train'
    mini_train_pipe: str = 'mini_train_pipe'
    mini_train_pipe_ho: str = 'mini_train_pipe_ho'
    pm_1: str = 'pm_1'
    live_trading: str = 'live_trading'


def scoped_dates(release_date: date, scope: ScopePrePost | str = ScopePrePost.all):
    if scope == ScopePrePost.pre:
        dates = [add_trade_days(release_date, i) for i in [-20, -15, -10, -5, -3, -2, -1, 0]]
    elif scope == ScopePrePost.train:
        dates = [add_trade_days(release_date, i) for i in [-20, -15, -10, -5, -3, -2, -1]]
    elif scope == ScopePrePost.post:
        dates = [add_trade_days(release_date, i) for i in [1, 2]]
    elif scope == ScopePrePost.mini_post:
        dates = [add_trade_days(release_date, i) for i in [1]]
    elif scope == ScopePrePost.mini_train:
        dates = [add_trade_days(release_date, i) for i in [-1, 0, 1]]
    elif scope == ScopePrePost.live_trading:
        dates = [add_trade_days(release_date, i) for i in [-1]]
    elif scope == ScopePrePost.mini_train_pipe:
        dates = [add_trade_days(release_date, i) for i in [-1, 0]]
    elif scope == ScopePrePost.mini_train_pipe_ho:
        dates = [add_trade_days(release_date, i) for i in [1]]
    elif scope == ScopePrePost.pm_1:
        dates = [add_trade_days(release_date, i) for i in [0, 1]]
    elif scope == ScopePrePost.all:
        dates = [add_trade_days(release_date, i) for i in [-20, -15, -10, -5, -3, -2, -1, 0, 1, 2]]
    else:
        dates = [add_trade_days(release_date, i) for i in [-20, -15, -10, -5, -3, -2, -1, 0, 1, 2]]
    return sorted(set(dates))