import os

from pathlib import Path

log_fn = 'log_{}.txt'
fp = Path(__file__)
src_path = fp.resolve().parents[1]

path_earnings = r'C:\repos\quantconnect\Lean\Algorithm.CSharp\Core\EarningsAnnouncements.json'


class Paths:
    """
    Project paths for easy reference.
    """
    project_path = src_path.parents[0]
    src_path = src_path
    common = os.path.join(src_path, 'shared')

    analytics = rf'C:\repos\quantconnect\Lean\Launcher\bin\Analytics'
    path_earnings = path_earnings if os.path.exists(path_earnings) else os.path.join(common, 'EarningsAnnouncements.json')
    path_models = Path(r'D:\trade\models')
