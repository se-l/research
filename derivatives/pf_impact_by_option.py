import pandas as pd

from copy import copy
from dataclasses import dataclass
from datetime import date
from logging import info
from typing import List

from connector.api_minlp.common import holdings2v_q, get_nlv_by_ds, get_weighted_dlnv, get_hedge_quantity, get_dPLuPLd, \
    get_utility_stats, calculate_utility, calc_d_abs_eq_position_hedged
from derivatives.earnings_release_ssvi import EarningsReleaseSimulation
from options.helper import timer
from options.types.earnings_config import EarningsConfig
from options.types.equity import Equity
from options.types.holding import Holding
from options.types.option import Option
from options.types.option_contract import OptionContract
from options.types.portfolio import Portfolio
from options.types.security import Security
from shared.constants import EarningsPreSessionDates


@dataclass
class PfImpact:
    option: str
    quantity: float
    dPL: float
    dPLuPLd: float
    utility: float
    quantityUnderlying: float
    dPLEqHedged: float
    dPLuPLdEqHedged: float
    holdings: List[Holding]
    iv_enter: float


class PfImpactByOption:
    def __init__(self, cfg: EarningsConfig, weight_util_q_eq: float):
        self.cfg = cfg
        self.weight_util_q_eq = weight_util_q_eq
        self.pf_impacts: List[PfImpact] = []
        self.pf = None
        self.er = None
        self.scoped_options = []
        self.m_dnlv01_estimated_buy, self.m_dnlv01_estimated_sell = None, None
        self.weighted_dnlv = None
        self.weighted_dnlv_up = None
        self.weighted_dnlv_down = None
        self._pf_0 = None
        self._pf_0_q_eq = None
        self._weighted_dnlv_0 = None
        self._dPLuPLd_0 = None
        self._equity = None

    """
        Refinement:
        - The ultimate short is exercised 1 day at EOD before release date. Also here can improve, research when the time value is high.
        Todo: Check how much the extrinsic value decreases towards release date. Want to short/long more than 1 D in advance?? IV goes up. Leaving money on the table?

        - Generalize. also investigate portfolios earning the term structure IV crush.

        # Questions:
        # - What's the optimal T-x days to sell near expiry options? Understands how far values change. How intrinsic value changes as release date comes up.
        # - Optimal near/long hedge ratios? Risk / Pnl Trade off... relate risk space and cost to hedge?
        """
    def set_pf0(self, pf: Portfolio):
        cfg = self.cfg
        self.pf = pf
        v_ds_ret = cfg.v_ds_ret

        self.er = EarningsReleaseSimulation(cfg)
        self.scoped_options = self.er.scoped_options
        self.m_dnlv01_estimated_buy = self.er.get_iv_transition_matrices().m_dnlv01_estimated_buy
        self.m_dnlv01_estimated_sell = self.er.get_iv_transition_matrices().m_dnlv01_estimated_sell

        v_q = holdings2v_q(pf, self.er.scoped_options)

        nlv_by_ds = get_nlv_by_ds(v_q, self.m_dnlv01_estimated_buy, self.m_dnlv01_estimated_sell)
        dct_nlv_by_ds = dict(zip(v_ds_ret, nlv_by_ds))
        dct_nlv_by_ds_up = dict([t for t in zip(v_ds_ret, nlv_by_ds) if t[0] > 0])
        dct_nlv_by_ds_down = dict([t for t in zip(v_ds_ret, nlv_by_ds) if t[0] < 0])

        self.weighted_dnlv = get_weighted_dlnv(dct_nlv_by_ds, cfg)
        self.weighted_dnlv_up = get_weighted_dlnv(dct_nlv_by_ds_up, cfg)
        self.weighted_dnlv_down = get_weighted_dlnv(dct_nlv_by_ds_down, cfg)

        return self

    # def get_hedge_quantity(self, pf=None):
    #     v_q = holdings2v_q(pf or self.pf, self.er.scoped_options)
    #     nlv_by_ds = get_nlv_by_ds(v_q, self.m_dnlv01_estimated_buy, self.m_dnlv01_estimated_sell)
    #     dct_nlv_by_ds = dict(zip(self.cfg.v_ds_ret, nlv_by_ds))
    #     return -get_delta_total_across_ds(dct_nlv_by_ds, self.cfg, self.er.s0)

    @property
    def equity(self):
        if self._equity is None:
            self._equity = Equity(self.cfg.sym)
        return self._equity


    @property
    def pf_0(self):
        if self._pf_0 is None:
            equity = Equity(self.cfg.sym)
            pf_0: Portfolio = copy(self.pf)
            self._pf_0 = pf_0
            self._pf_0_q_eq = pf_0[equity]
            self._weighted_dnlv_0, self._dPLuPLd_0 = get_utility_stats(pf_0, self.er.scoped_options,
                                                                       self.m_dnlv01_estimated_buy,
                                                                       self.m_dnlv01_estimated_sell, self.cfg)
        return self._pf_0

    @property
    def weighted_dnlv_0(self):
        if self._weighted_dnlv_0 is None:
            _ = self.pf_0
        return self._weighted_dnlv_0

    @property
    def dPLuPLd_0(self):
        if self._dPLuPLd_0 is None:
            _ = self.pf_0
        return self._dPLuPLd_0

    def add_pf_impact(self, o: Security, quantity: float):
        # Goal: reduce equity exposure and increase gains shorting IV.

        # Quantity the utility comparing a perfectly equity-hedged portfolio with current holdings of options and a
        # portfolio where we add 1 more option.
        equity = Equity(o.underlying_symbol)
        q_hedge_to_delta_zero = get_hedge_quantity(self.pf_0, scoped_options=self.er.scoped_options, m_dnlv01_buy=self.m_dnlv01_estimated_buy, m_dnlv01_sell=self.m_dnlv01_estimated_sell, cfg=self.cfg)
        q_eq_0_hedged = self.pf[equity] + q_hedge_to_delta_zero

        # For a reasonably count of 1k scoped options, this is 2k calls. bad
        for q in [quantity, -quantity]:
            pf_1 = copy(self.pf_0).add_holding(o, q)

            q_eq_1 = pf_1[equity]
            # pf_1.add_holding(equity, -pf_1[equity])
            weighted_dnlv_1, dPLuPLd_1 = get_utility_stats(pf_1, self.er.scoped_options, self.m_dnlv01_estimated_buy, self.m_dnlv01_estimated_sell, self.cfg)

            # subtract 0 from 1, to get the utility of adding the respective option
            dPL = weighted_dnlv_1 - self.weighted_dnlv
            delta_dPLuPLd = abs(self.dPLuPLd_0) - abs(dPLuPLd_1)  # how equally distributed my PnL is wrt to upward/downward move. Positive if _1 is smaller than _0

            # Only this value needs to compare perfectly hedged pfs - this is a slow call
            d_abs_eq_position = calc_d_abs_eq_position_hedged(equity, q_eq_0_hedged, pf_1, self.er.scoped_options, self.m_dnlv01_estimated_buy, self.m_dnlv01_estimated_sell, self.cfg)

            utility = calculate_utility(dPL, delta_dPLuPLd, d_abs_eq_position, self.weight_util_q_eq)

            if isinstance(o, Option):
                iv_enter = self.er.get_assumed_fill_iv(pf=Portfolio({o: q}), s0=self.er.s0, calc_date0=self.cfg.release_date)[o]
            else:
                iv_enter = 0

            self.pf_impacts.append(PfImpact(
                str(o.symbol) if o else '', q, 0, 0, utility, q_eq_1, dPL, delta_dPLuPLd, copy(pf_1).get_holdings(), iv_enter
            ))


@timer
def get_pf_impact_by_option(underlying: str, pf: Portfolio, release_date: date, weight_util_q_eq: float = 1) -> PfImpactByOption:
    """
    Given a universe of options and a current portfolio,
    generate table with following entries:
    weight_util_q_eq - is tricky to quantify. Less equity is meant to reduce the risk of imbalanced IV forecasts.
    """
    info(release_date)

    cfg = EarningsConfig(underlying, release_date, plot=True, plot_last=True,
                         moneyness_limits=(0.8, 1.2), abs_delta_limits=(0.1, 0.9), seq_ret_threshold=0.002,
                         min_tenor=0.0, max_tenor=1, add_equity_holdings=True, portfolio=pf, run_solver=False,
                         n_contracts=10, earnings_iv_drop_regressor_model_name_version='f_20260407-205754',
                         )
    inst = PfImpactByOption(cfg, weight_util_q_eq).set_pf0(pf)

    # q_hedge = get_hedge_quantity(pf, scoped_options=inst.er.scoped_options, m_dnlv01_buy=inst.m_dnlv01_estimated_buy, m_dnlv01_sell=inst.m_dnlv01_estimated_sell, cfg=cfg)
    # inst.add_pf_impact(Equity(underlying), q_hedge) # Current Portfolio

    for sec in inst.scoped_options:
        inst.add_pf_impact(sec, 1)

    return inst


# if __name__ == '__main__':
#     tickers = 'CSCO'
#     for take in [-2]:
#         for sym in tickers.split(','):
#             release_date = EarningsPreSessionDates(sym)[take]
#             pf = Portfolio({Option(OptionContract.from_ib_symbol(s), release_date): q for s, q in {
#                 "CSCO  260618C00077500": 1,
#                 "CSCO  270115P00077500": 2,
#                 "CSCO  260213C00070000": 1,
#                 "CSCO  260213C00072000": 1,
#                 "CSCO  260213C00073000": 1,
#                 "CSCO  260220C00073000": -1,
#                 "CSCO  260213C00076000": 1,
#                 "CSCO  260220C00077500": 1,
#                 "CSCO  260213P00086000": -2,
#                 "CSCO  260327C00095000": 1
#             }.items()})
#             pf.add_holding(Equity(sym), -272)
#             inst = get_pf_impact_by_option(sym, pf, release_date)
#             pd.DataFrame(inst.pf_impacts).sort_values('utility', ascending=False).to_excel(rf'D:\trade\tmp\pf_impacts_{sym}-ReleaseDate-{release_date}.xlsx', index=False)
#
#
