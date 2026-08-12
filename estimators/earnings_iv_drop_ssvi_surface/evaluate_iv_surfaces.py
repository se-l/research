from datetime import date

from derivatives.surfaces.evaluate_ivs_ssvi import plot_evaluation
from estimators.earnings_iv_drop_ssvi_surface.train_estimator import get_v_ivs, get_df_qt
from options.frame_builder import get_option_frame, available_sym_dates
from options.types.enums import Resolution
from options.types.equity import Equity


def evaluate_surfaces(equity: Equity):
    # dates = [date(2024, 9, 19)]
    dates = [i.date for i in available_sym_dates(equity.symbol.upper())]
    dates = [date(2024, 4,5)]
    resolution = Resolution.second
    seq_ret_threshold = 0.002
    seq_ret_threshold_surface = None
    arb_free = False
    calc_date = dates[0]
    # dividends = get_dividends(equity.symbol.upper(), calc_date)

    # option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
    # df_q, df_t = get_df_qt(option_frame, equity, dates)
    # samples = get_calibration_items(calc_date, df_q, dividends, side, df_t)
    # max_tenor = max((ci.tenor_dt for ci in samples))
    # ivs = IVSurface(equity)
    # ivs.calibrate_surface(equity, {max_tenor: [ci for ci in samples][0]}, calc_date)
    # plot_evaluation(ivs, df_q, calc_date)

    v_ivs = get_v_ivs(equity, dates, resolution, seq_ret_threshold=seq_ret_threshold, arb_free=arb_free, seq_ret_threshold_surface=seq_ret_threshold_surface)
    for ivs in v_ivs:
        # if ivs.v_calibrated_params is not None and len(ivs.v_calibrated_params) > 0:
        option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
        df_q, df_t = get_df_qt(option_frame, equity, dates)
        plot_evaluation(ivs, df_q, calc_date)

if __name__ == "__main__":
    """
    pending model improvements:
    - weight by tenor
    - IV of nearest neighbor stocks, sector and overall market
    - historical stock vol changes. so actual vol vs implied vol ( can be accomplished with daily data )
    """
    # Test calibration on the last tenor. Both IV and price are massively off mid-price/iv
    equities = [
        # Equity('PEP'),
        # Equity('FDX'),
        Equity('DAL'),
    ]
    for equity in equities:
        evaluate_surfaces(equity)
    print('Done.')
