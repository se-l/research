import os

from typing import List
from multiprocessing import Pool

from options.calibrate_yield_curve import calibrate_yield_curve_and_store
from options.helper import get_pkl_cache_key
from options.surfaces.processors import get_v_ivs
from options.types.scope_pre_post import ScopePrePost, scoped_dates
from options.types.quote_side import QuoteSide
from options.types.sym_date import SymDate
from shared.constants import EarningsPreSessionDates
from shared.modules.logger import info
from options.frame_builder import available_sym_dates, check_data_presence
from options.types.enums import Resolution
from options.types.equity import Equity
from shared.paths import Paths


def list_valid_training_sym_dates(tickers=None):
    tickers_in = ','.join(tickers or os.listdir(Paths.path_data.joinpath('option', 'usa', 'second'))).upper()

    root = Paths.path_analysis_frames
    clear_prefix = 'v_ivs'
    scope = ScopePrePost.mini_train
    arb_free = False
    seq_ret_threshold_surface = None
    seq_ret_threshold = 0.002

    exists = []
    exists_calibrated = []
    missing = []
    for sym in tickers_in.split(','):
        for release_date in EarningsPreSessionDates(sym):
            sym_date = SymDate(sym, release_date)
            if check_data_presence(sym_date, scope):
                if all(( os.path.exists(os.path.join(root, get_pkl_cache_key(clear_prefix, Equity(sym), [dt], Resolution.second, seq_ret_threshold, QuoteSide.mid, arb_free, seq_ret_threshold_surface))) for dt in scoped_dates(release_date, scope))):
                    exists_calibrated.append(sym_date)
                else:
                    exists.append(sym_date)
            else:
                missing.append(sym_date)

    info(f'Data exists: {len(exists)} - {", ".join(map(str,exists))}')
    info(f'')
    info(f'Data exists and IVS calibrated: {len(exists_calibrated)} - {", ".join(map(str,exists_calibrated))}')
    info(f'')
    info(f'Data missing: {len(missing)} - {", ".join(map(str, missing))}')


    info(f'')
    info(f'Data exists - IVS not calibrated: {len(exists)}')
    info(f'Data exists and IVS calibrated: {len(exists_calibrated)}')
    info(f'No Data: {len(missing)}')


def cache_all_surfaces(tickers: List[str]=None, arb_free=False):
    tickers_in = ','.join(tickers or os.listdir(Paths.path_data.joinpath('option', 'usa', 'second'))).upper()

    clear_prefix = 'v_ivs'
    root = Paths.path_analysis_frames
    scope = ScopePrePost.mini_train
    sym_dates = available_sym_dates(tickers_in, scope)
    seq_ret_threshold_surface = None

    payloads = []
    already_exist = 0
    for sym_date in sym_dates:
        sym = sym_date.symbol.lower()
        release_date = sym_date.date
        equity = Equity(sym)
        resolution = Resolution.second
        seq_ret_threshold = 0.002

        info(f'Processing {sym} on {release_date}...')
        for dt in scoped_dates(release_date, scope):
            fn = get_pkl_cache_key(clear_prefix, equity, [dt], resolution, seq_ret_threshold, QuoteSide.mid, arb_free, seq_ret_threshold_surface)
            if os.path.exists(os.path.join(root, fn)):
                already_exist += 1
            else:
                payloads += [(equity, [dt], resolution, seq_ret_threshold, arb_free, seq_ret_threshold_surface)]

    info(f'cache_all_surfaces(): Processing {len(payloads)} sym-dates. Skipping {already_exist} - already_exist')
    execute(payloads, already_exist)


def execute(payloads, already_exist):
    n_processes = 4
    info(f'get_ivs_mp: Processing {len(payloads)} payloads in {n_processes}. # already_exist: {already_exist}')
    with Pool(n_processes) as pool:
        pool.map(exec_payload, payloads)
    # for p in payloads:
    #     exec_payload(p)

def exec_payload(payload):
    equity = payload[0]
    seq_ret_threshold = payload[3]
    v_ivs = get_v_ivs(*payload)
    calc_date = payload[1][0]
    calibrate_yield_curve_and_store(v_ivs, calc_date, equity, seq_ret_threshold=seq_ret_threshold)
    # if not YieldCurve().has_calibrated_curve(calc_date, equity) and v_ivs:
    #     try:
    #         calibrate_yield_curve_and_store(v_ivs, calc_date, equity, seq_ret_threshold=seq_ret_threshold)
    #     except Exception as e:
    #         warning(e)
    info(f'yield curve calibrated for {equity.symbol} - {calc_date}')


if __name__ == "__main__":
    # FIXME - payload needs to become a type
    """
    pending model improvements:
    - weight by tenor
    - IV of nearest neighbor stocks, sector and overall market
    - historical stock vol changes. so actual vol vs implied vol ( can be accomplished with daily data )
    """
    # Test calibration on the last tenor. Both IV and price are massively off mid price/iv
    # equity = Equity('DAL')
    equities = [
        Equity('NKE').symbol.upper(),
        # Equity('DAL').symbol.upper(),
        # Equity('FDX').symbol.upper(),
        # Equity('TGT').symbol.upper(),
        # Equity('DELL').symbol.upper(),
        # Equity('PEP').symbol.upper(),
        # Equity('PATH').symbol.upper(),
        # Equity('ORCL').symbol.upper(),
    ]
    # equities = [Equity('FDX').symbol.upper(), Equity('DAL').symbol.upper()]

    cache_all_surfaces(equities)
    # cache_all_surfaces([])
    # list_valid_training_sym_dates(['NKE'])
    print('Done.')
