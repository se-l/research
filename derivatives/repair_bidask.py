import numpy as np
import pandas as pd
from arbitragerepair import constraints, repair

if __name__ == '__main__':
    # normalise strikes and prices
    normaliser = constraints.Normalise()
    normaliser.fit(T, K, C, F)
    T1, K1, C1 = normaliser.transform(T, K, C)

    _, _, C1_bid = normaliser.transform(T, K, C_bid)
    _, _, C1_ask = normaliser.transform(T, K, C_ask)

    # repair arbitrage - l1-norm objective
    epsilon1 = repair.l1(mat_A, vec_b, C1)

    # repair arbitrage - l1ba objective
    spread_ask = C1_ask - C1
    spread_bid = C1 - C1_bid
    spread = [spread_ask, spread_bid]

    epsilon2 = repair.l1ba(mat_A, vec_b, C1, spread=spread)