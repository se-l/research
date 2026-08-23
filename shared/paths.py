import os

from pathlib import Path
from dotenv import dotenv_values

fp = Path(__file__)
src_path = fp.resolve().parents[1]
cfg = dotenv_values(src_path.joinpath(".env"))

class Paths:
    """
    Project paths for easy reference.
    """
    project_path = src_path.parents[0]
    src_path = src_path
    common = os.path.join(src_path, 'shared')

    path_trade = Path(cfg['PATH_TRADE'])
    analytics = path_trade.joinpath('Analytics')
    path_models = path_trade.joinpath('models')
    path_data = path_trade.joinpath('data')
    path_ib = path_trade.joinpath('ib')

    path_data_alternative = path_data.joinpath('alternative')
    path_data_interest_rate = path_data_alternative.joinpath('interest-rate')
    path_symbol_properties = path_data.joinpath('symbol-properties')
    path_market_hours = path_data.joinpath('market-hours')
    path_activity_reports_ytd = path_ib.joinpath('activityReportsYTD')
    path_analysis_frames = analytics.joinpath('analysis_frames')
    path_calibration = analytics.joinpath('calibration')
    path_api_cache = analytics.joinpath('api_cache')

    path_earnings = path_symbol_properties.joinpath('EarningsAnnouncements.json')
    path_dividend_yields = path_symbol_properties.joinpath('DividendYields.json')
    path_market_hours_database = path_market_hours.joinpath('market-hours-database.json')


def mkdir(path: str | Path) -> str | Path:
    if not os.path.exists(path):
        os.makedirs(path)
    return path
