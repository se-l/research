import io
import os
import time
import pytz
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass, field
from zipfile import ZipFile
from itertools import groupby
from typing import List, Dict, Any
from pathlib import Path
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.common import BarData, HistoricalTickBidAsk, HistoricalTick, HistoricalTickLast
from datetime import datetime, timedelta, date
from shared.constants import IP, PORT, dt_fmt_eastern, n_ticks_requested, TZ_USEASTERN, dt_fmt_ymd, dt_fmt_ib_bar, TZ_UTC, dt_fmt_iso
from connector.ib.contract import make_contract
from connector.ib.enums import TradeType, WhatToShow, Exchange, Resolution, IBSecType
from connector.ib.typess.QuoteBar import QuoteBar
from connector.ib.typess.TickTrade import TickTrade
from connector.ib.typess.TickBidAsk import TickBidAsk
from connector.ib.typess.TradeBar import TradeBar
from options.helper import is_holiday
from options.types.enums import SecurityType
from shared.modules.logger import info, warning, error

map_right = {'C': 'call', 'P': 'put'}
map_trade_type = {
    WhatToShow.TRADES: TradeType.trade,
    WhatToShow.BID_ASK: TradeType.quote,
    WhatToShow.BID: 'quote_bid',
    WhatToShow.ASK: 'quote_ask',
    WhatToShow.HISTORICAL_VOLATILITY: 'hist_vol',
}
bp = 10_000
map_sec2exchange_dflt = {
    IBSecType.STK: Exchange.SMART,
    IBSecType.OPT: Exchange.SMART,
}


@dataclass
class Task:
    resolution: Resolution | str
    sec_type: IBSecType | str
    start: datetime
    end: datetime
    whatToShow: str
    symbol: str
    exchange: str = None
    market: str = 'usa'
    contracts: List[Contract] = field(default_factory=list)  # to be filled on the fly
    endDateTimeBar: str = None

    def __post_init__(self):
        self.endDateTimeBar: str = self.end.strftime(dt_fmt_ib_bar)
        self.exchange = map_sec2exchange_dflt.get(self.sec_type)
        if self.symbol in ('CSCO', 'FANG'):
            self.exchange = Exchange.ISLAND
        # if self.symbol in ('BMW',):
        #     self.exchange = Exchange.IBIS

        if self.symbol in ('BMW',):
            self.market = 'europe'

    @property
    def dt(self):
        return datetime(self.start.year, self.start.month, self.start.day)


# @dataclass
# class Request:
#     id: int
#     task: Task
#     contract: Contract
#     reqId: int
#     startDateTime: str
#     endDateTime: str
#     whatToShow: str
#     durationStr: str


class IBClient(EWrapper, EClient):
    """
    Pacing Violation1 occurs whenever one or more of the following restrictions is not observed:
    Making identical historical data requests within 15 seconds.
    Making six or more historical data requests for the same Contract, Exchange and Tick Type within two seconds.
    Making more than 60 requests within any ten minute period.
    """
    def __init__(self, tasks: List[Task]):
        EClient.__init__(self, self)
        self.tasks = tasks
        self.reqId2Task: Dict[int, Task] = {}
        self.reqId2Contract: Dict[int, Contract] = {}
        self.contract2Ticks: Dict[Contract, List[Any]] = defaultdict(list)
        self.contract2Records: Dict[Contract, List[Any]] = defaultdict(list)

        self.error_msg = ''
        self.pending_request_ids = set()
        self.request_cache = {}

    def connect(self, host, port, clientId):
        super().connect(host, port, clientId)
        time.sleep(.5)
        return self

    def register_new_req_id(self, task: Task, contract: Contract) -> int:
        id_ = max(self.pending_request_ids) + 1 if self.pending_request_ids else 0
        self.pending_request_ids.add(id_)
        self.reqId2Task[id_] = task
        self.reqId2Contract[id_] = contract
        return id_

    def remove_req_id(self, reqId: int):
        if reqId in self.pending_request_ids:
            self.pending_request_ids.remove(reqId)
        self.request_cache.pop(reqId, None)
        # self.reqId2Task.pop(reqId)
        # self.reqId2Contract.pop(reqId)
        info(f'{len(self.pending_request_ids)} requests pending')
        if not self.pending_request_ids:
            self.disconnect()

    def error(self, reqId, errorCode, errorString):
        if errorCode == 162 and 'No data returned for a contract' in errorString:  # ... Only applies to bars, not ticks...
            self.error_msg = f"Error {reqId} - {errorCode}: {errorString}"
            error(self.error_msg)
            # The error breaks whole event loop. Remove the contract from the list and continue.
            self.save_records(self.reqId2Task[reqId], self.reqId2Contract[reqId])
            # self.remove_req_id(reqId)
        else:
            super().error(reqId, errorCode, errorString)

    def historicalTicks(self, reqId: int, ticks: List[HistoricalTick], done: bool):
        """returns historical tick data when whatToShow=MIDPOINT"""
        for tick in ticks:
            info("HistoricalTick. ReqId:", reqId, tick)

    def historicalTicksLast(self, reqId: int, ticks: List[HistoricalTickLast], done: bool):
        """returns historical tick data when whatToShow=BID_ASK"""
        contract: Contract = self.reqId2Contract[reqId]
        task: Task = self.reqId2Task[reqId]

        info(f"{datetime.now()} historicalTicksLast: reqId: {reqId} - {contract.localSymbol}. Fetched # Ticks: {len(ticks)}" if ticks else "")
        for tick in ticks:
            self.contract2Ticks[contract].append(TickTrade(time=tick.time, trade_sale=int(tick.price * bp), volume=tick.size, exchange=tick.exchange))

        if len(ticks) == 0 or \
                len(ticks) < n_ticks_requested or \
                (len(self.contract2Ticks[contract]) > 0 and self.contract2Ticks[contract][-1].time > task.end.timestamp()):
            self.save_ticks(reqId, contract)
            self.contract2Ticks.pop(contract)
        else:
            # time.sleep(10)  # Avoid pacing violation. Making more than 60 requests within any ten minute period.
            new_start_time = datetime.fromtimestamp(self.contract2Ticks[contract][-1].time + 1).astimezone(pytz.timezone(TZ_USEASTERN))
            newReqId = self.register_new_req_id(task, contract)
            info(f'{newReqId} - Requesting {contract.localSymbol} from {new_start_time} {task.whatToShow}')
            self.fetch_ticks_by_contract(reqId=newReqId, startDateTime=new_start_time)

        self.remove_req_id(reqId)

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        info(f'reqId {reqId} | ContractDetails: {contractDetails}')
        task = self.reqId2Task[reqId]
        task.contracts.append(contractDetails.contract)

    def contractDetailsEnd(self, reqId):
        # All contract details have been received
        task = self.reqId2Task[reqId]
        info(f"ReqId: {reqId} - Received {len(task.contracts)} contracts")

        if task.contracts:
            """Remove contracts that have already been save to .zip file"""
            try:
                zip_path = get_file_path_zip(task, task.dt.strftime(dt_fmt_ymd), mkdir=True)
                with ZipFile(zip_path.absolute(), 'r') as zp:
                    contracts = [c for c in task.contracts if get_csv_nm(task, task.dt, c) not in zp.namelist()]
                    info(f"Received {len(task.contracts)} filtered down to {len(contracts)} Contracts.")
                    task.contracts = contracts
            except FileNotFoundError:
                pass
            if task.contracts:
                newReqId = self.register_new_req_id(task, task.contracts[0])
                if task.resolution == Resolution.tick:
                    self.fetch_ticks_by_task(newReqId)
                else:
                    self.fetch_bars_by_task(newReqId)
        else:
            warning(f"ReqId: {reqId} - No contracts found for {task.dt.strftime(dt_fmt_ymd)}.")

        self.remove_req_id(reqId)

    def fetch_ticks_by_contract(self, reqId: int, startDateTime: datetime = None):
        task = self.reqId2Task[reqId]

        if reqId == 21 and self.reqId2Contract[reqId].symbol == 'FAST' and startDateTime.date() == date(2023, 9, 27) and task.whatToShow == WhatToShow.TRADES:
            warning(f'Subtracting a second from startDateTime for {self.reqId2Contract[reqId].symbol} on {startDateTime.date()}')
            startDateTime = startDateTime - timedelta(seconds=1)

        startTime = (startDateTime or task.start).strftime(dt_fmt_eastern)
        endTime = task.end.strftime(dt_fmt_eastern)
        info(f'{reqId} - Requesting {task.symbol} from {startTime} to {endTime} {task.whatToShow}')

        request = dict(
            reqId=reqId,
            contract=self.reqId2Contract[reqId],
            startDateTime=startTime,
            endDateTime=endTime,
            numberOfTicks=n_ticks_requested,
            useRth=0,
            ignoreSize=True,
            miscOptions=[],
            whatToShow=task.whatToShow
        )
        self.request_cache[reqId] = request
        # time_requested = datetime.now()

        self.reqHistoricalTicks(**request)

        # while reqId in self.pending_request_ids:
        #     # async sleep
        #     asyncio.sleep(1)
        #     if (datetime.now() - time_requested).seconds > 30:
        #         warning(f"Request {reqId} took too long. Cancelling and resending the request subtracting a second from startDateTime.")
        #         contract = self.reqId2Contract[reqId]
        #         self.remove_req_id(reqId)
        #
        #         newReqId = self.register_new_req_id(task, contract)
        #         new_start_time = startDateTime - timedelta(seconds=1)
        #         info(f'{newReqId} - Requesting {contract.localSymbol} from {new_start_time}')
        #         self.fetch_ticks_by_contract(reqId=newReqId, startDateTime=new_start_time)

    def fetch_ticks_by_task(self, reqId: int):
        task = self.reqId2Task[reqId]
        info(f'Fetching ticks for {len(task.contracts)} contracts {task.whatToShow}.')
        for contract in task.contracts:
            newReqId = self.register_new_req_id(task, contract)
            self.fetch_ticks_by_contract(newReqId, task.start)
        self.remove_req_id(reqId)

    def historicalTicksBidAsk(self, reqId: int, ticks: List[HistoricalTickBidAsk], done: bool):
        """returns historical tick data when whatToShow=BID_ASK"""
        contract: Contract = self.reqId2Contract.get(reqId)
        task: Task = self.reqId2Task.get(reqId)
        if contract is None or task is None:
            warning(f"Contract or Task not found for reqId: {reqId}")
            return

        info(f"{datetime.now()} {reqId} - {contract.localSymbol}. Fetched #ticks: {len(ticks)}" if ticks else "")
        for tick in ticks:
            # HistoricalTickBidAsk. ReqId: 1 Time: 1681740021, TickAttriBidAsk: BidPastLow: 0, AskPastHigh: 0, PriceBid: 0.950000, PriceAsk: 1.050000, SizeBid: 248, SizeAsk: 1
            # Resolution: Second, not every update. Presumably Last state.
            self.contract2Ticks[contract].append(TickBidAsk(time=tick.time, bid_sale=tick.priceBid * bp, bid_size=tick.sizeBid, ask_sale=tick.priceAsk * bp, ask_size=tick.sizeAsk))

        if len(ticks) == 0 or \
                len(ticks) < n_ticks_requested or \
                (len(self.contract2Ticks[contract]) > 0 and self.contract2Ticks[contract][-1].time > task.end.timestamp()):
            self.save_ticks(reqId, contract)
            self.contract2Ticks.pop(contract)
        else:
            # time.sleep(10)  # Avoid pacing violation. Making more than 60 requests within any ten minute period.
            new_start_time = datetime.fromtimestamp(self.contract2Ticks[contract][-1].time + 1).astimezone(pytz.timezone(TZ_USEASTERN))
            newReqId = self.register_new_req_id(task, contract)
            self.fetch_ticks_by_contract(reqId=newReqId, startDateTime=new_start_time)

        self.remove_req_id(reqId)

    def historicalData(self, reqId: int, bar: BarData):
        """
        OHLCV has different meaning depending on what to show.
        https://interactivebrokers.github.io/tws-api/historical_bars.html#hd_request
        """
        contract: Contract = self.reqId2Contract[reqId]
        task: Task = self.reqId2Task[reqId]
        # info(f"historicalData: {datetime.now()} {reqId} - {contract.localSymbol}. Received #bars: {len(bar)}" if bar else "")

        if task.whatToShow == WhatToShow.TRADES:
            self.contract2Records[contract].append(TradeBar(time=int(bar.date), open=bar.open * bp, high=bar.high * bp, low=bar.low * bp, close=bar.close * bp, volume=bar.volume))
        elif task.whatToShow in (WhatToShow.BID, WhatToShow.ASK, WhatToShow.BID_ASK):
            self.contract2Records[contract].append(QuoteBar(time=int(bar.date), open=bar.open * bp, high=bar.high * bp, low=bar.low * bp, close=bar.close * bp, volume=0))
        else:  # Hist volatility
            self.contract2Records[contract].append(QuoteBar(time=int(bar.date), open=bar.open, high=bar.high, low=bar.low, close=bar.close, volume=0))

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        contract: Contract = self.reqId2Contract[reqId]
        info(f"HistoricalDataEnd: Contract={contract.symbol}, ReqId={reqId}, FromUTC={start}, ToUTC={end}")

        task = self.reqId2Task[reqId]
        bars = self.contract2Records[contract]
        if task.resolution in (Resolution.daily,):
            min_dt_est = datetime.strptime(str(min(bars, key=lambda x: x.time).time), dt_fmt_ymd)
        else:
            min_dt_int = min(bars, key=lambda x: x.time).time
            min_dt_est = datetime.fromtimestamp(min_dt_int).astimezone(pytz.timezone(TZ_USEASTERN))

        # Consider adding here additional requests like in ticks until all specified in task have been received. Then save to file.
        if min_dt_est.hour < 8 \
            or task.resolution in (Resolution.daily,):
            self.save_records(task, contract)
            self.contract2Records.pop(contract)
        else:
            # time.sleep(0.5)  # Avoid pacing violation. Making more than 60 requests within any ten minute period.
            new_end_time = datetime.fromtimestamp(min_dt_int - 1).astimezone(pytz.timezone(TZ_UTC))
            newReqId = self.register_new_req_id(task, contract)
            info(f'ReqId={newReqId} - Requesting {contract.localSymbol} up to {new_end_time}')
            self.fetch_bars_by_contract(reqId=newReqId, endDateTime=new_end_time)

        self.remove_req_id(reqId)

    # def tickOptionComputation(self, reqId, tickType, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice, *args, **kwargs):
    #     info(
    #         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Implied Volatility: {impliedVol}, Delta: {delta}, Option Price: {optPrice}, Gamma: {gamma}, Vega: {vega}, Theta: {theta}")
    #     return impliedVol, delta, optPrice, gamma, vega, theta

    def fetch_bars_by_task(self, reqId: int):
        task = self.reqId2Task[reqId]
        info(f'Fetching bars for {len(task.contracts)} contracts.')
        for contract in task.contracts:
            newReqId = self.register_new_req_id(task, contract)
            if task.resolution == Resolution.second:
                self.fetch_bars_by_contract(newReqId, task.end, durationStr='2000 S')
            elif task.resolution == Resolution.daily:
                self.fetch_bars_by_contract(newReqId, task.end, durationStr='365 D')

        self.remove_req_id(reqId)

    def fetch_bars_by_contract(self, reqId: int, endDateTime: datetime, durationStr: str='2000 S'):
        task = self.reqId2Task[reqId]
        map_resolution2bar_size = {
            Resolution.second: '1 secs',
            Resolution.minute: '1 min',
            Resolution.hour: '1 hour',
            Resolution.daily: '1 day',
        }

        self.reqHistoricalData(
            reqId=reqId,
            contract=self.reqId2Contract[reqId],
            endDateTime=endDateTime.strftime(dt_fmt_ib_bar),  # yyyymmdd HH:mm:ss ttt
            durationStr=durationStr,  # task.durationStr,  # Pacing violation limit 2000 sec
            barSizeSetting=map_resolution2bar_size[task.resolution],
            useRTH=0,
            chartOptions=[],
            formatDate=2,
            whatToShow=task.whatToShow,
            keepUpToDate=False,
        )

    def save_records(self, task: Task, contract: Contract):
        records: List[TradeBar] | List[QuoteBar] = self.contract2Records[contract]
        records = sorted(records, key=lambda x: x.time)
        # remove duplicate timestamps for bars. could overlap..
        if not records:
            zip_path = get_file_path_zip(task, task.dt.strftime(dt_fmt_ymd), mkdir=True)
            csv_nm = get_csv_nm(task, task.dt, contract)

            with ZipFile(zip_path.absolute(), 'a') as zp:
                # delete the file if it exists
                if csv_nm not in zp.namelist():
                    zp.writestr(csv_nm, '')
            info(f"Saved {csv_nm} in {zip_path.name}. 0 records.")
        else:
            for dt, gp_record in groupby(records, lambda x: datetime.fromtimestamp(x.time).astimezone(pytz.timezone(TZ_USEASTERN)).date()):
                buffer = io.BytesIO()
                # gp_record = [1678714200]
                gp_record = list(gp_record)
                dt_tz = datetime.fromtimestamp(gp_record[0].time, tz=pytz.UTC).astimezone(pytz.timezone(TZ_USEASTERN))
                sec_till_midnight = int(dt_tz.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
                # datetime(dt.year, dt.month, dt.day, tzinfo=pytz.timezone(TZ_USEASTERN))
                # (gp_record[0] - sec_till_midnight) / (60**2)  # Expect 9.5. Start of NYC trading

                df = pd.DataFrame(gp_record)
                df['time'] = (df['time'] - sec_till_midnight) * 1000

                df = df[~df['time'].duplicated()]

                if task.whatToShow == WhatToShow.TRADES:
                    df = df[df['volume'] > 0]

                df.to_csv(buffer, header=False, index=False)
                zip_path = get_file_path_zip(task, dt.strftime(dt_fmt_ymd), mkdir=True)
                csv_nm = get_csv_nm(task, dt, contract)

                with ZipFile(zip_path.absolute(), 'a') as zp:
                    if csv_nm not in zp.namelist():
                        zp.writestr(csv_nm, buffer.getvalue())
                info(f"Saved {csv_nm} in {zip_path.name}. {len(gp_record)} records.")

        self.contract2Records[contract] = []

    def save_ticks(self, reqId, contract: Contract = None):
        task = self.reqId2Task[reqId]
        contract = contract or self.reqId2Contract[reqId]

        ticks: List[TickTrade] | List[TickBidAsk] = self.contract2Ticks[contract]
        ticks = [t for t in ticks if task.start.timestamp() < t.time < task.end.timestamp()]

        if not ticks:
            zip_path = get_file_path_zip(task, task.dt.strftime(dt_fmt_ymd), mkdir=True)
            csv_nm = get_csv_nm(task, task.dt, contract)

            with ZipFile(zip_path.absolute(), 'a') as zp:
                # delete the file if it exists
                if csv_nm not in zp.namelist():
                    zp.writestr(csv_nm, '')
            info(f"Saved {csv_nm} in {zip_path.name}. 0 Ticks.")
        else:
            for dt, gp_records in groupby(ticks, lambda x: datetime.fromtimestamp(x.time).astimezone(pytz.timezone(TZ_USEASTERN)).date()):
                buffer = io.BytesIO()
                # gp_records = [1678714200]
                gp_records = list(gp_records)
                dt_tz = datetime.fromtimestamp(gp_records[0].time, tz=pytz.UTC).astimezone(pytz.timezone(TZ_USEASTERN))
                sec_till_midnight = int(dt_tz.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
                # datetime(dt.year, dt.month, dt.day, tzinfo=pytz.timezone(TZ_USEASTERN))
                # (gp_records[0] - sec_till_midnight) / (60**2)  # Expect 9.5. Start of NYC trading

                df = pd.DataFrame(gp_records)
                df['time'] = (df['time'] - sec_till_midnight) * 1000
                df.to_csv(buffer, header=False, index=False)
                zip_path = get_file_path_zip(task, dt.strftime(dt_fmt_ymd), mkdir=True)
                csv_nm = get_csv_nm(task, dt, contract)

                with ZipFile(zip_path.absolute(), 'a') as zp:
                    if csv_nm not in zp.namelist():
                        zp.writestr(csv_nm, buffer.getvalue())
                info(f"Saved {csv_nm} in {zip_path.name}. {len(gp_records)} records.")

    def make_requests(self):
        for task in self.tasks:
            contract = get_contract(task)
            reqId = self.register_new_req_id(task, contract)
            # self.reqContractDetails(reqId, contract)
            # continue
            if task.sec_type == IBSecType.STK:
                task.contracts = [contract]
                if task.resolution == Resolution.tick:
                    self.fetch_ticks_by_task(reqId)
                else:
                    self.fetch_bars_by_task(reqId)
            else:
                self.reqContractDetails(reqId, contract)


def get_contract(task: Task) -> Contract:
    return make_contract(symbol=task.symbol, secType=task.sec_type, exchange=task.exchange, currency="EUR")


def get_file_path_zip(task: Task, date: str, mkdir: bool = False) -> Path:
    if task.sec_type == IBSecType.OPT:
        zip_path = Path(os.path.join(Paths.path_data, SecurityType.option, task.market, task.resolution, task.symbol.lower(), f'{date}_{map_trade_type[task.whatToShow]}_american.zip'))
    elif task.sec_type == IBSecType.STK:
        zip_path = Path(os.path.join(Paths.path_data, SecurityType.equity, task.market, task.resolution, task.symbol.lower(), f'{date}_{map_trade_type[task.whatToShow]}.zip'))
    else:
        raise ValueError(f'Unknown sec_type: {task.sec_type}')

    if mkdir:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
    return zip_path


def csv_member_exists(zip_path: Path, csv_nm: str) -> bool:
    if os.path.exists(zip_path):
        with ZipFile(zip_path, 'r') as zp:
            return csv_nm in zp.namelist()
    return False


def get_csv_nm(task: Task, dt: datetime.date, contract: Contract) -> str:
    if contract.secType == IBSecType.OPT:
        return f'{dt.strftime(dt_fmt_ymd)}_{contract.symbol.lower()}_{task.resolution}_{map_trade_type[task.whatToShow]}_american_{map_right[contract.right]}_{int(contract.strike * bp)}_{contract.lastTradeDateOrContractMonth}.csv'
    elif contract.secType == IBSecType.STK:
        return f'{dt.strftime(dt_fmt_ymd)}_{contract.symbol.lower()}_{map_trade_type[task.whatToShow]}_{task.resolution}.csv'
    else:
        raise ValueError(f'Unknown sec_type: {contract.secType}')


def fetch(symbols: List[str], start: str, end: str, what_to_show, sec_type: IBSecType | str, resolution: Resolution | str, port: int = PORT):
    tasks = []
    for symbol in symbols:
        for start_dt in pd.date_range(start=start, end=end, freq='1 D'):
            if is_holiday(start_dt):
                continue
            dt = datetime(start_dt.year, start_dt.month, start_dt.day)
            start_dt = pytz.timezone(TZ_USEASTERN).localize(dt)
            end_dt = pytz.timezone(TZ_USEASTERN).localize(dt + timedelta(hours=23, minutes=59, seconds=59))
            task = Task(resolution=resolution, sec_type=sec_type, start=start_dt, end=end_dt, whatToShow=what_to_show, symbol=symbol)
            contract = get_contract(task)
            csv_nm = get_csv_nm(task, dt, contract)
            if csv_member_exists(get_file_path_zip(task, dt.strftime(dt_fmt_ymd)), csv_nm):
                warning(f'{symbol} {dt.strftime(dt_fmt_ymd)} already exists. Delete before re-fetching.')
                continue
            tasks.append(task)
    if tasks:
        app = IBClient(tasks).connect(IP, port, clientId=1)
        app.make_requests()
        app.run()
    info(f'Done Fetching {symbols} {start}-{end}.')

def fetch_daily_trades(symbols: List[str], start: str, end: str, what_to_show, sec_type: IBSecType | str, resolution: Resolution | str, port: int = PORT):
    """
    Fetching IB equity trades with daily resolution in a single trask
    Not yet saving correctly...
    """
    tasks = []
    for symbol in symbols:
        start_dt = pytz.timezone(TZ_USEASTERN).localize(datetime.strptime(start, dt_fmt_iso))
        end_dt = pytz.timezone(TZ_USEASTERN).localize(datetime.strptime(end, dt_fmt_iso))
        task = Task(resolution=resolution, sec_type=sec_type, start=start_dt, end=end_dt, whatToShow=what_to_show, symbol=symbol)
        tasks.append(task)

    if tasks:
        app = IBClient(tasks).connect(IP, port, clientId=1)
        app.make_requests()
        app.run()
    info(f'Done Fetching {symbols} {start}-{end}.')



# def clean_up(min_date: str, tags: List[str], path=r'C:\repos\trade\data\option\usa'):
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file[:8] >= min_date \
#                     and any((t in file for t in tags)) \
#                     and any(res in root for res in ['minute'])\
#                     :
#                 info(f'Deleting {os.path.join(root, file)}')
#                 os.remove(os.path.join(root, file))


def daily_run():
    start = '2023-06-06'
    end = '2023-06-06'
    # improvement. dont group requests by dates just to know when to create an empty file. Create empty for any data for which no ticks were received except for weekends.
    # Cuts amount of requests by n days. A must once loading a lot... Continue daily yday loads for warm up as usual..
    fetch(
        symbols=['HPE', 'IPG', 'AKAM', 'AOS', 'A', 'MO', 'FL', 'ALL', 'ARE', 'ZBRA', 'AES', 'APD', 'ALLE', 'LNT', 'ZTS', 'ZBH'],
        start=start, end=end, what_to_show=WhatToShow.BID_ASK, sec_type=IBSecType.OPT, resolution=Resolution.tick
    )
        # upsample(sec_type=IBSecType.OPT, market='usa', resolution_from='minute', resolution_to='daily', symbol=symbol.lower(), trade_type='quote_american')
        # gen_openinterest_files(sec_type=IBSecType.OPT, market='usa', resolution=Resolution.minute, symbol=symbol.lower())
        # gen_openinterest_files(sec_type=IBSecType.OPT, market='usa', resolution=Resolution.daily, symbol=symbol.lower())


if __name__ == "__main__":
    # daily_run()
    # EQUITY TRADES TICKS
    # start = '2000-03-25'
    # end = '2020-03-27'
    start = '2026-03-13'
    end = '2026-03-13'
    for symbol in ['DAL']:
        # fetch(symbol=symbol, start=datetime.today().isoformat(), end=datetime.today().isoformat(), what_to_show=WhatToShow.TRADES, sec_type=IBSecType.STK, resolution=Resolution.daily, durationStr='180 D')
        # fetch(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.TRADES, sec_type=IBSecType.STK, resolution=Resolution.minute)
        # fetch(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.BID_ASK, sec_type=IBSecType.STK, resolution=Resolution.minute)
        # EQUITY BID ASK
        # fetch_daily_trades(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.TRADES, sec_type=IBSecType.STK, resolution=Resolution.daily)
        # fetch(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.BID_ASK, sec_type=IBSecType.STK, resolution=Resolution.second)
        # fetch(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.TRADES, sec_type=IBSecType.STK, resolution=Resolution.second)
        fetch(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.BID_ASK, sec_type=IBSecType.STK, resolution=Resolution.tick)
        # fetch(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.BID, sec_type=IBSecType.STK, resolution=Resolution.tick)
        # fetch(symbols=[symbol], start=start, end=end, what_to_show=WhatToShow.ASK, sec_type=IBSecType.STK, resolution=Resolution.tick)

        # OPTION TRADES TICKS
        # fetch(symbol=symbol, start=start, end=end, what_to_show=WhatToShow.TRADES, sec_type=IBSecType.OPT, resolution=Resolution.minute)
        # OPTION BID ASK
        # fetch(symbol=symbol, start=start, end=end, what_to_show=WhatToShow.BID, sec_type=IBSecType.OPT, resolution=Resolution.minute)
        # fetch(symbol=symbol, start=start, end=end, what_to_show=WhatToShow.ASK, sec_type=IBSecType.OPT, resolution=Resolution.minute)

        # EQUITY HISTORICAL VOLATILITY
        # fetch(start='2023-04-15', end='2023-04-15', what_to_show=WhatToShow.HISTORICAL_VOLATILITY, sec_type=IBSecType.STK, resolution=Resolution.daily)

        # merge equity BID ASK into single Quote file...
        # merge_option_quote_bid_ask(IBSecType.OPT, 'usa', 'minute', symbol.lower())

    # The daily run.
    # end = start = (datetime.now() - timedelta(days=1)).date().isoformat()  # '2023-04-01'

    info('Done.')
