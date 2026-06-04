using Dates

mutable struct EarningsConfig
    sym::String
    release_date::Date
    plot::Bool
    plot_last::Bool
    max_scoped_options::Int
    resolution::Resolution
    seq_ret_threshold::Float64
    min_tenor::Float64
    max_tenor::Float64
    moneyness_limits::NTuple{2, Float64}
    abs_delta_limits::NTuple{2, Float64}
    add_equity_holdings::Bool
    v_dIVT1::Vector{Float64}
    earnings_iv_drop_regressor::Union{AbstractEarningsIVDropRegressor, Nothing}  # adjust type if needed
    ts_time_power::Float64
    v_ds_ret::Vector{Float64}
    portfolio::Portfolio
    run_solver::Bool
    solver_t_params::NTuple{4, Float64}
    n_contracts::Int
    earnings_iv_drop_regressor_model_name_version::String
    var_neg_ratio::Float64
    weight_max_t_curve::Float64
    weight_wing_lift::Float64
    sub_portfolios_threshold::Float64

    function EarningsConfig(
        sym::String,
        release_date::Date;
        plot::Bool = true,
        plot_last::Bool = true,
        max_scoped_options = 400,
        resolution::Resolution = resolution_minute,
        seq_ret_threshold = 0.002,
        min_tenor = 0.0,
        max_tenor = 999.0,
        moneyness_limits::NTuple{2, Float64} = (0.75, 1.25),
        abs_delta_limits::NTuple{2, Float64} = (0.05, 0.95),
        add_equity_holdings::Bool = true,
        v_dIVT1::Vector{Float64} = collect(0.0:-0.04/1:-0.04),  # linspace(0.0, -0.04, 2)
        earnings_iv_drop_regressor::Union{AbstractEarningsIVDropRegressor, Nothing} = nothing,
        ts_time_power = 0.5,
        v_ds_ret::Vector{Float64} = collect(range(0.8, 1.2, length=21)),
        portfolio::Portfolio = Portfolio(),
        run_solver::Bool = true,
        solver_t_params::NTuple{4, Float64} = (2.46172191, 2.7678077519107513, -0.058245258591748644, 5.577267709644770),
        n_contracts = 20,
        earnings_iv_drop_regressor_model_name_version::String = "f_20260407-205754",
        var_neg_ratio = 1.5,
        weight_max_t_curve = 5.0,
        weight_wing_lift = 0.2,
        sub_portfolios_threshold=0.98,
    )
        new(
            sym, release_date, plot, plot_last, max_scoped_options,
            resolution, seq_ret_threshold, min_tenor, max_tenor,
            moneyness_limits, abs_delta_limits, add_equity_holdings,
            v_dIVT1, earnings_iv_drop_regressor, ts_time_power,
            v_ds_ret, portfolio, run_solver, solver_t_params,
            n_contracts, earnings_iv_drop_regressor_model_name_version,
            var_neg_ratio, weight_max_t_curve, weight_wing_lift, sub_portfolios_threshold
        )
    end
end

function Base.hash(cfg::EarningsConfig, h::UInt)::UInt
    portfolio_str = join((string(k, v) for (k, v) in cfg.portfolio), ",")
    return hash((
        cfg.sym, cfg.release_date, cfg.plot, cfg.plot_last,
        cfg.max_scoped_options, cfg.resolution, cfg.seq_ret_threshold,
        cfg.min_tenor, cfg.max_tenor, cfg.moneyness_limits,
        cfg.abs_delta_limits, cfg.add_equity_holdings, cfg.ts_time_power,
        portfolio_str, cfg.run_solver, cfg.solver_t_params, cfg.n_contracts,
    ), h)
end

function get_earnings_cfg(sym::String, release_date::Date)::EarningsConfig
    return EarningsConfig(sym, release_date;
        plot = false,
        plot_last = false,
        moneyness_limits = (0.8, 1.2),
        abs_delta_limits = (0.1, 0.9),
        min_tenor = 0.0,
        max_tenor = 1.0,
        add_equity_holdings = false,
    )
end