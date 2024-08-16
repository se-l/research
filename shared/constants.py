import os
import datetime
import json
import json5

from shared.modules.logger import error
from shared.paths import Paths
file_root = Paths.path_data

try:
    with open(Paths.path_dividend_yields, 'r') as fh:
        DividendYield = json5.load(fh)
except Exception as e:
    error("No DividendYield file found", e)
    DividendYield = {}

try:
    with open(Paths.path_earnings, 'r') as fh:
        earnings_announcements = json.load(fh)
except Exception as e:
    error("No EarnningsAnnouncement file found", e)
    earnings_announcements = []

DiscountRateMarket = (DividendYield or {}).get('DiscountRateMarket', 0.0435)  # https://www.bloomberg.com/markets/rates-bonds/government-bonds/us


def EarningsPreSessionDates(sym: str):
    return sorted([datetime.date.fromisoformat(x['Date']) for x in earnings_announcements if x['Symbol'].upper() == sym.upper()])


model_nm_earnings_iv_drop_regressor = os.environ.get('model_nm_earnings_iv_drop_regressor', 'earnings_iv_drop_regressor_v3_2024-07-08.joblib')
