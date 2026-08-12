class Exchange:
    SMART = "SMART"
    ARCA = "ARCA"
    NASDAQ = "NASDAQ"
    ISLAND = "ISLAND"
    IBIS = "IBIS"


class IBSecType:
    STK: str = 'STK'
    OPT: str = 'OPT'


class ExchangeCode:
    A = "NYSE MKT Stock Exchange"
    B = "NASDAQ OMX BX Stock Exchange"
    C = "National Stock Exchange"
    D = "FINRA"
    I = "International Securities Exchange"
    J = "Direct Edge A Stock Exchange"
    K = "Direct Edge X Stock Exchange"
    M = "Chicago Stock Exchange"
    N = "New York Stock Exchange"
    T = "NASDAQ OMX Stock Exchange"
    P = "NYSE Arca SM"
    S = "Consolidated Tape System"
    TQ = "NASDAQ Stock Exchange"
    W = "CBOE Stock Exchange"
    X = "NASDAQ OMX PSX Stock Exchange"
    Y = "BATS Y-Exchange"
    Z = "BATS Exchange"


class WhatToShow:
    TRADES = "TRADES"
    MIDPOINT = "MIDPOINT"
    BID_ASK = "BID_ASK"
    BID = "BID"
    ASK = "ASK"
    HISTORICAL_VOLATILITY = "HISTORICAL_VOLATILITY"


class Resolution:
    tick = 'tick'
    second = 'second'
    minute = 'minute'
    hour = 'hour'
    daily = 'daily'


class BarSize:
    secs1 = '1 secs'
    secs5 = '5 secs'
    secs10 = '10 secs'
    secs15 = '15 secs'
    secs30 = '30 secs'
    min1 = '1 min'
    mins2 = '2 mins'
    mins3 = '3 mins'
    mins5 = '5 mins'
    mins10 = '10 mins'
    mins15 = '15 mins'
    mins20 = '20 mins'
    mins30 = '30 mins'
    hour1 = '1 hour'
    hours2 = '2 hours'
    hours3 = '3 hours'
    hours4 = '4 hours'
    hours8 = '8 hours'
    day1 = '1 day'
    week1 = '1 week'
    month1 = '1 month'


map_resolution2bar_size = {
    'second': BarSize.secs1,
    'minute': BarSize.min1,
    'hour': BarSize.hour1,
    'daily': BarSize.day1,
}


class TradeType:
    quote = "quote"
    trade = "trade"
    iv_quote = "iv_quote"
    iv_trade = "iv_trade"
