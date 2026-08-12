import pandas as pd


class IEstimate:
    """Estimate side by:
    - Loading label ranges from inlux
    - Samples are events where series diverges from expectation: load from inlux
    - Weights: Less unique sample -> lower weight
    - CV. Embargo area
    - Store in Experiment
    - Generate feature importance plot
    - ToInflux: Signed estimates
    """
    # What consumbers need to find
    ex = None
    boosters = None
    df = None
    df_ho = None
    pred_label_val = None
    pred_label_ho = None
    ps_label = None
    ps_label_ho = None

    # def __init__(self, exchange: Exchange, sym, start: datetime, end: datetime, labels=None, signals=None, features=None):
    #     self.exchange = exchange
    #     self.sym = sym
    #     self.start = start
    #     self.end = end
    #     self.labels = labels
    #     self.signals = signals
    #     self.features = features
    #     self.window_aggregator_window = [int(2**i) for i in range(20)]
    #     self.window_aggregator_func = ['sum']
    #     self.window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, self.window_aggregator_func)]
    #     self.boosters = []
    #     self.tags = {}
    #     self.ex = ex(sym)
    #     logger.info(self.ex)
    #     self.df = None

    def load_label(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series): pass
    @staticmethod
    def curtail_nnan_front_end(df: pd.DataFrame) -> pd.DataFrame: pass
    def load_sample_from_signals(self, df: pd.DataFrame) -> pd.Index: pass
    def apply_embargo(self): pass
    def calc_weights(self): pass
    def load_features(self) -> pd.DataFrame: pass
    @staticmethod
    def exclude_too_little_data(df: pd.DataFrame) -> pd.DataFrame: pass
    @staticmethod
    def exclude_non_stationary(df: pd.DataFrame) -> pd.DataFrame: pass
    def assemble_frames(self): pass
