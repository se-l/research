include(joinpath(@__DIR__, "../../init.jl"))

using Dates
using .Fino

# Settings
const n_processes = 24
const market = "usa"

"""
    massive_data_dl_docker(security_type::SecurityType, start_dt::Date, stop_dt::Date, tickers::String; n_clients=256, flush_interval=1000) -> String

Generate a docker command for downloading massive data.
"""
function massive_data_dl_docker(security_type::SecurityType, start_dt::Date, stop_dt::Date, tickers::String; n_clients=256, flush_interval=1000)::String
    sec_type_str = if security_type == security_type_equity
        "Equity"
    elseif security_type == security_type_option
        "Option"
    else
        error("Unsupported SecurityType: $security_type")
    end

    # Format dates as YYYYMMDD
    start_str = Dates.format(start_dt, "yyyymmdd")
    stop_str = Dates.format(stop_dt, "yyyymmdd")
    stop_str = "20260604"

    cmds = [
        "cd C:\\repos\\quantconnect\\Lean; ",
        "echo \$pwd;",
        "docker compose run ",
        "--rm ",
        "-e app=pdl ",
        "-e from-date=$(start_str)-00:00:00 ",
        "-e to-date=$(stop_str)-00:00:00 ",
        "-e tickers=$(tickers) ",
        "-e market=usa ",
        "-e resolution=Tick ",
        "-e security-type=$(sec_type_str) ",
#        "-e skip-filled= ",
        # f'-e skip-empty= '
        # f'-e skip-modified-since=2024-09-14 '
        "-e n-clients=$(n_clients) ",
        "-e flush-interval=$(flush_interval) ",
        "toolbox;"
    ]
    return join(cmds, "")
end

"""
    ps_exec(command::String)

Execute a PowerShell command.
"""
function ps_exec(command::String)
    # Using run() to call powershell
    try
        # Powershell expects the command as a single string if using -Command
        # We need to escape double quotes if any, but here we probably don't have many.
        run(`powershell -ExecutionPolicy ByPass -Command $command`)
    catch e
        @error "Error executing command: $command" exception=e
    end
end

"""
    download_single_symbol(symbol::String, date::Date, security_type::SecurityType; n_clients=32)

Download a single file for a given symbol, date, and security type.
"""
function download_single_symbol(symbol::String, date::Date, security_type::SecurityType; n_clients=32)
    cmd = massive_data_dl_docker(security_type, date, date, symbol; n_clients=n_clients)
    @info "Downloading single symbol: $symbol, Date: $date, Type: $security_type"
    ps_exec(cmd)
end

function download(security_type::SecurityType, v_ticker; n_days_lookback, n_clients=32, takes=nothing)
    configs = get_configs(v_ticker; n_days_lookback=n_days_lookback, takes=takes)
    v_commands = [massive_data_dl_docker(security_type, cfg.start, cfg.stop, cfg.tickers; n_clients=n_clients) for cfg in configs]

    @info "# download requests $(security_type): $(length(v_commands))"
    # To limit concurrency to 8:
    ch = Channel{Int}(8)
    @sync for cmd in v_commands
        put!(ch, 1)
        @async begin
            try
                ps_exec(cmd)
            finally
                take!(ch)
            end
        end
    end
end


function main()
    v_ticker = ["CRM", "MRVL", "SNOW"]
    v_ticker = ["CRWD"]

    download(security_type_equity, v_ticker; n_days_lookback=-20, n_clients=32, takes=[-1])
    download(security_type_option, v_ticker; n_days_lookback=-1, n_clients=128, takes=[-1])

    @info "Done."
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
    sym = "CRWD"
#    download_single_symbol(sym, Date(2026, 6, 3), security_type_equity)
#    download_single_symbol(sym, Date(2026, 6, 3), security_type_option)
    # afterwards
end
