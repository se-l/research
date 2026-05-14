import pickle
import time
from datetime import datetime, date
from pprint import pprint

from options.frame_builder import get_option_frame
from options.surfaces.processors import get_df_qt, get_calibration_items, scope_df
from options.types.dividend import get_dividends
from options.types.enums import Resolution
from options.types.equity import Equity
from options.types.iv_surface_essvi import IVSurface
from options.types.option import get_price_cuda
from options.types.quote_side import QuoteSide


class TestIVSurfaceEssvi:
    def test_compare_cpp_jl_ivs_calibration(self):
        # [DELL - 2023-08-31,
        #  DELL - 2023-11-30,
        #  DELL - 2024-02-29,
        #  DELL - 2024-05-30,
        #  DELL - 2024-08-29,
        #  DELL - 2025-11-25]
        equity = Equity("DELL")
        n_samples_per_tenor = 1000
        dates = [date(2024, 8, 29)]
        side = QuoteSide.mid
        resolution = Resolution.second
        seq_ret_threshold = 0.002

        option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
        df_q, df_t = get_df_qt(option_frame, equity, dates)

        start = datetime(1970, 1, 1)
        end = datetime(2099, 1, 1)
        calc_date = df_q.index.get_level_values('ts')[0].date()
        df_q, df_t = scope_df(df_q, start, end), scope_df(df_t, start, end)

        dividends = get_dividends(equity.symbol.lower(), start=calc_date)

        ivs = IVSurface(equity)
        ivs_fp_jl = IVSurface(equity)
        ivs_jl = IVSurface(equity)
        calibration_items = get_calibration_items(calc_date, df_q, dividends, side, df_t)

        # Fit entire surface
        samples_all = {item.tenor_dt: item for item in calibration_items}
        samples = {item.tenor_dt: item.downsample(min(n_samples_per_tenor, len(item.price))) for item in calibration_items}
        n_samples_all = sum([len(ci.price) for ci in samples_all.values()])
        n_samples = sum([len(ci.price) for ci in samples.values()])
        print(f'df_qt2surface(): Downsampled from {n_samples_all} to {n_samples}')

        # t0 = time.time()
        # ivs.calibrate_surface(equity, samples, calc_date, x0_in={})
        # print(f'iv_surface(): Calibration took {time.time() - t0} seconds')
        #
        # t0 = time.time()
        # ivs_fp_jl.calibrate_surface(equity, samples, calc_date, x0_in={})
        # print(f'iv_surface_fp_jl(): Calibration took {time.time() - t0} seconds')

        t0 = time.time()
        ivs_jl.calibrate_surface_jl(equity, samples, calc_date, x0_in={})
        print(f'iv_surface_jl(): Calibration took {time.time() - t0} seconds')

        print(ivs.v_calibrated_params)
        print(ivs_fp_jl.v_calibrated_params)
        print(ivs_jl.v_calibrated_params)

        # with open(r'D:\ivs.pkl', 'wb') as f:
        #     pickle.dump(ivs, f)
        # with open(r'D:\ivs_fp_jl.pkl', 'wb') as f:
        #     pickle.dump(ivs_fp_jl, f)
        with open(r'D:\ivs_jl.pkl', 'wb') as f:
            pickle.dump(ivs_jl, f)

        # print("Percentage differences (ivs_jl vs ivs reference):")
        # for i in range(len(ref)):
        #     if ref[i] != 0.0:
        #         pct_diff = ((jl[i] - ref[i]) / ref[i]) * 100.0
        #         print(f"Element {i}: {pct_diff}%")
        #     else:
        #         print(f"Element {i}: Cannot compute (reference is zero)")
        #
        #
