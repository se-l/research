from dataclasses import dataclass


@dataclass
class Holding:
    symbol: str  # To be refactored. A str is too error prone. Needs to be a Security object.
    quantity: int

    @classmethod
    def from_holding_pb(cls, holding_pb: object):
        return cls(holding_pb.symbol, holding_pb.quantity)

    def __hash__(self):
        return hash(self.symbol)
