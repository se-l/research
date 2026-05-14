using PyCall
using Dates
using Distributed

include("RawDataConfig.jl")

# ── Bootstrap PyCall to use your conda env ──────────────────────────────────
const PYTHON_EXE = raw"C:\Users\seb\miniconda3\envs\py314\python.exe"
ENV["PYTHON"] = PYTHON_EXE

# ── Python imports ───────────────────────────────────────────────────────────
const connector_constants = pyimport("connector.constants")
const raw_data_processors = pyimport("connector.raw_data_processors")
const options_helper = pyimport("options.helper")
const options_enums = pyimport("options.types.enums")
const shared_constants = pyimport("shared.constants")
const shared_logger = pyimport("shared.modules.logger")

const dt_fmt_ymd = connector_constants.dt_fmt_ymd
const ps_exec = raw_data_processors.ps_exec
const add_trade_days = options_helper.add_trade_days
const SecurityType = options_enums.SecurityType
const EarningsPreSessionDates = shared_constants.EarningsPreSessionDates

const N_PROCESSES = 24
const MARKET = "usa"

# ── massive_data_dl_docker ───────────────────────────────────────────────────
function massive_data_dl_docker(
security_type,
start::Date,
stop::Date,
tickers;
n_clients::Int = 32,
)::Vector{String}
    start_str = Dates.format(start, "yyyy-mm-dd")
    stop_str = Dates.format(stop, "yyyy-mm-dd")

    cmds = [
        raw"cd C:\repos\quantconnect\Lean; ",
        raw"echo $pwd;",
        "docker compose run " *
        "--rm " *
        "-e app=pdl " *
        "-e from-date=$(start_str)-00:00:00 " *
        "-e to-date=$(stop_str)-00:00:00 " *
        "-e tickers=$(tickers) " *
        "-e market=usa " *
        "-e resolution=Tick " *
        "-e security-type=$(titlecase(string(security_type))) " *
        "-e skip-filled= " *
        # "-e skip-empty= "                          *
        # "-e skip-modified-since=2024-09-14 "       *
        "-e n-clients=$(n_clients) " *
        "toolbox;",
    ]
    return [join(cmds, " ")]
end

# ── get_configs (primary overload) ───────────────────────────────────────────
function get_configs(
v_ticker::Vector{String};
n_days_lookback::Int = -1,
n_days_lookahead::Int = 2,
)
    configs = []
    for take in -30:-1
        for ticker in v_ticker
            local release_date
            try
                release_date = EarningsPreSessionDates(ticker)[take + 1] # 1-based in Python list
            catch e
                continue
            end
            start = add_trade_days(release_date, n_days_lookback)
            stop = add_trade_days(release_date, n_days_lookahead)
            push!(configs, RawDataConfig(start, stop, ticker))
        end
    end
    return configs
end

# ── get_configs (secondary overload) ───────────────────────────────────────────
function get_configs(
v_ticker::Vector{String},
release_date::Date;
n_days_lookback::Int = -1,
n_days_lookahead::Int = 2,
)
    configs = []
    for ticker in v_ticker
        start = add_trade_days(release_date, n_days_lookback)
        stop = add_trade_days(release_date, n_days_lookahead)
        push!(configs, RawDataConfig(start, stop, ticker))
    end
    return configs
end

# ── main ─────────────────────────────────────────────────────────────────────
function main()
    v_ticker = [
        # "NVDA",
        "PEP", "PGR", "FAST", "CAG", "JPM", "UNH", "WFC", "C", "BLK", "STT", "SCHW",
        "PLD", "BK", "ASML", "NFLX", "GS", "ELV", "TSM", "ABT", "MMC", "AXP", "CDNS", "TMUS",
        "HSY", "CBRE", "PG", "ON", "MRK", "PFE", "PSA", "O", "VRSK", "AMGN", "MRNA",
        "LNG", "XPO", "PLTR", "TSN", "ONON", "TJX", "TGT", "JD", "AMAT", "ROST",
        "BIDU", "DKS", "ADI", "SNOW", "RY", "WDAY", "MRVL", "DLTR", "ULTA", "PDD",
        "HPE", "CRM", "CRWD", "AVGO", "DELL", "DG", "MDB", "PATH", "DOCU", "ORCL",
        "SNX", "MU", "KMX", "JBL", "CCL", "EXC", "PANW", "CSCO", "FANG", "ADBE",
        # "FDX", "NKE", "DAL",
    ]
    v_ticker = ["DAL"]

    v_config_eq = get_configs(v_ticker; n_days_lookback = -20)
    v_config_op = get_configs(v_ticker; n_days_lookback = -1)

    v_commands = Vector{Vector{String}}()
for cfg in v_config_eq
    push!(v_commands, massive_data_dl_docker(SecurityType.equity, cfg.start, cfg.stop, cfg.tickers))
end
for cfg in v_config_op
    push!(v_commands, massive_data_dl_docker(SecurityType.option, cfg.start, cfg.stop, cfg.tickers))
end

# Mirrors: multiprocessing.Pool(min(1, len(v_commands))).map(ps_exec, v_commands)
n_workers = min(1, length(v_commands))
if n_workers > 0
    addprocs(n_workers)
    pmap(ps_exec, v_commands)
end

shared_logger.info("Done.")
end

main()