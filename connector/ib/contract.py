from ibapi.contract import Contract

from connector.ib.enums import Exchange


def make_contract(symbol: str, secType: str, exchange: str = Exchange.SMART, currency: str = "USD", **kwargs) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = secType
    contract.exchange = exchange
    contract.currency = currency
    for k, v in kwargs.items():
        setattr(contract, k, v)
    return contract
