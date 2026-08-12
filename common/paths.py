import os

from pathlib import Path

log_fn = 'log_{}.txt'
fp = Path(__file__)
src_path = fp.resolve().parents[1]


class Paths:
    """
    Project paths for easy reference.
    """
    project_path = src_path.parents[0]
    src_path = src_path
    trader = os.path.join(src_path, 'trader')
    files = os.path.join(src_path, 'files')
    dir_norm_tal = os.path.join(src_path, 'norm_tal')
    layers = os.path.join(src_path, 'layers')
    config = os.path.join(src_path, 'config')
    layer_bars = os.path.join(layers, 'bars')
    layer_settings = os.path.join(layer_bars, 'settings.yaml')

    data = os.path.join(project_path, 'data')
    qc_forex = os.path.join(data, 'forex')
    qc_crypto = os.path.join(data, 'crypto')
    qc_bitfinex_crypto = os.path.join(qc_crypto, 'bitfinex')
    qc_bitmex_crypto = os.path.join(qc_crypto, 'bitmex')
    bitmex_raw = os.path.join(qc_bitmex_crypto, 'raw')
    bitfinex_tick = os.path.join(qc_bitfinex_crypto, 'tick')
    bitmex_raw_online_quote = r'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/quote/{}.csv.gz'
    bitmex_raw_online_trade = r'https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/{}.csv.gz'

    trade_ini_fn = os.path.join(src_path, 'trade_ini.json')
    model_features = os.path.join(project_path, 'model', 'model_features.json')
    trade_model = os.path.join(project_path, 'model', 'supervised')
    backtests = os.path.join(project_path, 'model', 'backtests')
    simulate = os.path.join(project_path, 'model', 'simulate')
    model_rl = os.path.join(project_path, 'model', 'reinforcement')
    path_buffer = os.path.join(project_path, 'subprocess_buffer')

    path_config_supervised = 'trader.train.config.supervised'
    path_config_reinforced = 'trader.train.config.reinforced'
    path_config_backtest = 'trader.backtest.config'
    path_config_labeler = 'trader.train.config.labeler'
    path_config_simulate = 'trader.data_loader.config.simulate'
