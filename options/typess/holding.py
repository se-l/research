from dataclasses import dataclass
from datetime import date

from options.typess.enums import SecurityType
from options.typess.option_contract import OptionContract
from options.typess.security import Security


@dataclass
class Holding:
    symbol: Security
    quantity: float

    @classmethod
    def from_holding_pb(cls, holding_pb: object,  calculation_date: date = None, security_type: SecurityType = None):  # protobuf holding_pb
        # recursive import
        from options.typess.equity import Equity
        from options.typess.option import Option

        security_type = security_type or SecurityType.infer_from_ib_symbol(holding_pb.symbol)

        if security_type == SecurityType.equity:
            return cls(Equity(holding_pb.symbol), holding_pb.quantity)
        elif security_type == SecurityType.option:
            return cls(Option(OptionContract.from_ib_symbol(holding_pb.symbol), calculation_date), holding_pb.quantity)
        else:
            raise ValueError(f'Unknown security type: {security_type}')

    def __hash__(self):
        return hash((self.symbol, self.quantity))
