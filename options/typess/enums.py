class OptionStyle:
    american = 'american'
    european = 'european'

    @classmethod
    def from_string(cls, s):
        if s in ('american',):
            return cls.american
        elif s in ('european',):
            return cls.european
        else:
            raise ValueError(f'Unknown option style: {s}')


class OptionRight:
    call = 'call'
    put = 'put'

    @classmethod
    def from_string(cls, s):
        if s in ('C', 'call'):
            return cls.call
        elif s in ('P', 'put'):
            return cls.put
        else:
            raise ValueError(f'Unknown option right: {s}')


class TickType:
    trade = 'trade'
    quote = 'quote'
    iv_quote = 'iv_quote'
    iv_trade = 'iv_trade'
    volume = 'volume'

    @classmethod
    def from_string(cls, s):
        if s in ('trade',):
            return cls.trade
        elif s in ('quote',):
            return cls.quote
        else:
            raise ValueError(f'Unknown tick type: {s}')


class TickDirection:
    up = 1
    down = -1

    @classmethod
    def from_string(cls, s):
        if s in ('up',):
            return cls.up
        elif s in ('down',):
            return cls.down
        else:
            raise ValueError(f'Unknown tick direction: {s}')


class SecurityType:
    equity = 'equity'
    option = 'option'

    @classmethod
    def from_string(cls, s):
        if s in ('equity',):
            return cls.equity
        elif s in ('option',):
            return cls.option
        else:
            raise ValueError(f'Unknown security type: {s}')


class GreeksEuOption:
    delta: str = 'delta'
    vega: str = 'vega'
    theta: str = 'theta'
    gamma: str = 'gamma'


class Resolution:
    daily = 'daily'
    hour = 'hour'
    minute = 'minute'
    second = 'second'
    tick = 'tick'


class CsvHeader:
    trade = ['time', 'open', 'high', 'low', 'close', 'volume']
    quote = ['time', 'bid_open', 'bid_high', 'bid_low', 'bid_close', 'bid_size', 'ask_open', 'ask_high', 'ask_low', 'ask_close', 'ask_size']
    # iv_quote = ['ask_iv', 'ask_price', 'bid_iv', 'bid_price', 'time', 'mid_price_underlying', ]
    # iv_trade = ['trade_iv', 'price', 'time', 'mid_price_underlying', ]
    iv_quote = ['time', 'mid_price_underlying', 'bid_price', 'bid_iv', 'ask_price', 'ask_iv', 'delta_bid', 'delta_ask']
    iv_trade = ['time', 'mid_price_underlying', 'price', 'trade_iv', 'delta']


class OptionPricingModel:
    CoxRossRubinstein = 'CoxRossRubinstein'
    AnalyticEuropeanEngine = 'AnalyticEuropeanEngine'
    FdBlackScholesVanillaEngine = 'FdBlackScholesVanillaEngine'


class SkewMeasure:
    ThirdMoment = 'ThirdMoment'
    Delta25Delta50 = 'Delta25Delta50'
    Delta25Delta25 = 'Delta25Delta25'
    M90M100 = 'M90M100'
    M90M110 = 'M90M110'
    M100M110 = 'M100M110'


class Market:
    usa = 'usa'
