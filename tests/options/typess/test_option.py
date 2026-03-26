import numpy as np

from datetime import date
from options.types.option import get_price_cuda
from shared.yield_curve import ZeroCurveData


class TestOption:
    def test_get_price_cuda_bad_ivs(self):
        """
        Testing
        - intrinsic value is returned for 0 and neg. vol for no r, q.
        - nan is returned for nan input IVs.
        """
        s = [100, 100, 100, 100, 100, 100]
        k = [90, 90, 90, 110, 110, 110]
        t = [1, 1, 1, 1, 1, 1]
        v_is_call = [1, 1, 1, 0, 0, 0]
        v_iv = [0, -1, np.nan, 0, -1, np.nan]

        expected = np.array([10, 10, np.nan, 10, 10, np.nan])

        for cpu in (False, True):
            res = get_price_cuda(
                s=np.array(s),
                k=np.array(k),
                t=np.array(t),
                v_is_call=np.array(v_is_call),
                iv=np.array(v_iv),
                dividends=[],
                calculation_date=date(2026, 1, 1),
                yield_curve=ZeroCurveData(times=[0, 99], rates=[0, 0]),
                cpu=cpu
            )

            np.testing.assert_array_equal(res, expected)