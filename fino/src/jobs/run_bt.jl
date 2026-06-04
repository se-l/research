include(joinpath(@__DIR__, "../init.jl"))

using Fino
using Dates
using PythonCall

function ps_docker_backtest(start_dt::Date, end_dt::Date, ticker::String; backtesting_holdings=nothing, deamon=true)
    start_iso = string(start_dt)
    end_iso = string(end_dt)
    container_name = "bt.dev.$(replace(lowercase(ticker), "," => "." )).$start_iso.$end_iso"
    time_str = Dates.format(now(), "HHMMSS")

    commands = [
        "cd C:\\repos\\quantconnect\\Lean;",
        "echo \$pwd;",
        "docker compose run " *
        "-e StartDate=$start_iso " *
        "-e EndDate=$end_iso " *
        "-e Ticker=$ticker " *
        "-e CONTAINER_NAME=$container_name " *
        "-e PricerProtocol=tcp " *
        "-e PricerHost=ws.dev " *
        "-e PricerPort=8102 "
    ]

    if !isnothing(backtesting_holdings)
        push!(commands, "-e BacktestingHoldings=\"$backtesting_holdings\" ")
    end

    push!(commands, "--name $(container_name)T$time_str bt.dev.1;")

    cmd_str = join(commands, " ")
    println(cmd_str)

    # In Julia, we can use run() to execute powershell commands
    # To mimic ps_exec(commands, deamon)
    ps_cmd = `powershell -ExecutionPolicy ByPass -Command $cmd_str`
    
    if deamon
        # Run in background and detach
        run(ps_cmd, wait=false)
    else
        run(ps_cmd)
    end
end

function earnings_target_holdings_bt(sym::String, take::Int; dct_holdings=nothing)
    dates = EarningsPreSessionDates(sym)
    idx = take < 0 ? length(dates) + take + 1 : take
    release_date = dates[idx]
    start_dt = release_date + Day(1)
    end_dt = release_date + Day(1)

    pf_holdings = if isnothing(dct_holdings)
        # We use run_on_py_thread to call the python version since it might be easier
        # and Fino already has it initialized.
        run_on_py_thread() do
            ensure_py_initialized_fino()
            py_derivatives_er = pyimport("derivatives.earnings_release")
            
            # Construct python EarningsConfig
            py_cfg = py_earnings_config_mod.EarningsConfig(
                sym, 
                pydate(year(release_date), month(release_date), day(release_date)),
                plot=true, 
                plot_last=true,
                earnings_iv_drop_regressor_model_name_version=model_nm_earnings_iv_drop_regressor,
                moneyness_limits=(0.8, 1.2), 
                abs_delta_limits=(0.1, 0.9),
                min_tenor=0.0, 
                add_equity_holdings=true
            )
            
            py_pf = py_derivatives_er.get_earnings_release_pf(py_cfg)
            
            if pyconvert(Bool, py_pf == pybuiltins.None) || pyconvert(Bool, pybuiltins.len(py_pf) == 0)
                return nothing
            end
            
            # Convert python dict/portfolio to a string representation
            # Python script did: '{' + ','.join([f'{sec}:{q}' for sec, q in pf_holdings.items()]) + '}'
            # Then some weird replacement.
            # Actually, Lean expects a JSON-like string for BacktestingHoldings.
            
            items = []
            for (sec, q) in py_pf.items()
                push!(items, "\"$(pyconvert(String, pybuiltins.str(sec)))\":$(pyconvert(Float64, q))")
            end
            "{" * join(items, ",") * "}"
        end
    else
        # If dct_holdings is provided (Julia Dict), convert to string
        items = []
        for (sec, q) in dct_holdings
            push!(items, "\"$sec\":$q")
        end
        "{" * join(items, ",") * "}"
    end

    if isnothing(pf_holdings)
        println("No suitable portfolio was found for $sym take=$take.")
        return
    end

    # The python script did some extra escaping:
    # pf_holdings_str = "'" + pf_holdings_str.replace("{", "{\"").replace(":", "\":").replace(",", ",\"") + "'"
    # My construction above already includes quotes for keys.
    
    ps_docker_backtest(start_dt, end_dt, sym, backtesting_holdings=pf_holdings)
end

function earnings_bt(sym::String, take::Int)
    dates = EarningsPreSessionDates(sym)
    idx = take < 0 ? length(dates) + take + 1 : take
    release_date = dates[idx]
    start_dt = add_trade_days(release_date, 0)
    end_dt = add_trade_days(release_date, 1)
    ps_docker_backtest(start_dt, end_dt, sym, deamon=true)
end

function main()
    v_args = [
#        ("CSCO", -2),
#        ("HPE", -3),
        ("CRWD", -2),
    ]
    for (sym, take) in v_args
        earnings_bt(sym, take)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
