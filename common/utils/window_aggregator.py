import re


class WindowAggregator:
    def __init__(self, window: str, aggregator: str):
        self.window = window
        self.aggregator = aggregator

    def to_dict(self):
        return {'window': self.window, 'aggregator': self.aggregator}

    def __repr__(self):
        return f'window-{self.window}-aggregator-{self.aggregator}'

    @classmethod
    def from_str(cls, repr: str):
        match = re.search(r'window-(\w.*)-aggregator-(\D)', repr)
        if match:
            return cls(match.group(1), match.group(2))
