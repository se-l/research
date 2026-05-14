# earnings_release_simulation.jl

using PythonCall

# ============================================================================
# Abstract interface
# ============================================================================

abstract type AbstractEarningsReleaseSimulation end

# ============================================================================
# Return-type wrappers
# ============================================================================

"""
    IVTransitionMatrices

Thin wrapper around the Python `IVTransitionMatrices` namedtuple/object.
Fields are accessed directly as Julia properties.
"""
struct IVTransitionMatrices
    m_dnlv01_worst_buy::Any
    m_dnlv01_worst_sell::Any
    m_dnlv01_best_buy::Any
    m_dnlv01_best_sell::Any
    m_dnlv01_estimated_buy::Any
    m_dnlv01_estimated_sell::Any
    v_delta0_mid::Any
end

"""
    SolverResult

Thin wrapper around the Python `SolverResult` namedtuple/object.
"""
struct SolverResult
    er::AbstractEarningsReleaseSimulation
    pf::Any
    pyo_model::Any
    pyo_inst::Any
end

# ============================================================================
# Python wrapper
# ============================================================================

"""
    PyEarningsReleaseSimulation(cfg, scoped_symbols)

Wraps the Python `EarningsReleaseSimulation` via PyCall.
"""
mutable struct PyEarningsReleaseSimulation <: AbstractEarningsReleaseSimulation
    py_obj::Py
    cfg::Any     # EarningsConfig (kept for Julia-side access)
end

function PyEarningsReleaseSimulation(cfg::EarningsConfig, scoped_symbols::Union{Vector{String}, Nothing}=nothing)
    py_cls = py_ers_mod.EarningsReleaseSimulation

    # Convert Julia EarningsConfig fields to Python types explicitly
    py_cfg = py_cls.__new__(py_cls)  # don't call __init__ yet

    # Build a Python-compatible config by passing a converted object
    # Simplest: pass individual fields as a Python namedtuple or dataclass
    py_cfg = py_earnings_config_mod.EarningsConfig(
        pyconvert(String, cfg.sym),
        pydate(year(cfg.release_date), month(cfg.release_date), day(cfg.release_date));
        plot                = cfg.plot,
        plot_last           = cfg.plot_last,
        moneyness_limits    = pytuple(cfg.moneyness_limits),
        abs_delta_limits    = pytuple(cfg.abs_delta_limits),
        min_tenor           = Float64(cfg.min_tenor),
        max_tenor           = Float64(cfg.max_tenor),
        add_equity_holdings = cfg.add_equity_holdings,
        n_contracts         = Int(cfg.n_contracts),
    )

    py_obj = py_cls(py_cfg, scoped_symbols=scoped_symbols === nothing ? nothing : pylist(scoped_symbols))
    return PyEarningsReleaseSimulation(py_obj, cfg)
end

# -- Property accessors --

function scoped_options(er::PyEarningsReleaseSimulation)
    return er.py_obj.scoped_options
end

function Base.getproperty(er::PyEarningsReleaseSimulation, sym::Symbol)
    if sym === :py_obj || sym === :cfg
        return getfield(er, sym)
    end
    if sym === :scoped_options
        return scoped_options(er)
    end
    # Delegate everything else to Python
    return getproperty(getfield(er, :py_obj), sym)
end

# -- Method wrappers --

"""
    set_ivs0_params(er, params)

Set initial IV surface parameters from protobuf params.
"""
function set_ivs0_params(er::PyEarningsReleaseSimulation, params)
    er.py_obj.set_ivs0_params(params)
    return er
end

function pf_to_py_pf(pf::Portfolio, calc_date)
    # Build a Python Portfolio from Julia Portfolio
    py_portfolio_mod = pyimport("options.types.portfolio")
    py_equity_mod_local = pyimport("options.types.equity")
    py_option_contract_mod = pyimport("options.types.option_contract")
    py_option_mod = pyimport("options.types.option")
    pydate = pyimport("datetime").date

    py_holdings = pydict()
    for h in pf
        sym = string(h.symbol)  # Julia → IB symbol string
        if contains(sym, " ")   # option
            py_contract = py_option_contract_mod.OptionContract.from_ib_symbol(sym)
            py_sec = py_option_mod.Option(py_contract, pydate(year(calc_date), month(calc_date), day(calc_date)))
        else                    # equity
            py_sec = py_equity_mod_local.Equity(sym)
        end
        py_holdings[py_sec] = h.quantity
    end
    return py_portfolio_mod.Portfolio(py_holdings)
end

"""
    get_assumed_fill_iv(er, pf, s0, calc_date) -> Dict

Get assumed fill implied volatilities for a portfolio.
"""
function get_assumed_fill_iv(er::PyEarningsReleaseSimulation, pf::Py, s0, calc_date)
    result = er.py_obj.get_assumed_fill_iv(pf, s0, calc_date)
    # Convert Python dict to Julia Dict{String, Float64}
    return Dict{String, Float64}(pyconvert(String, pybuiltins.str(k)) => pyconvert(Float64, v) for (k, v) in result.items())
end

function get_assumed_fill_iv(er::PyEarningsReleaseSimulation, pf::Portfolio, s0, calc_date)
    py_pf = pf_to_py_pf(pf, calc_date)
    py_calc_date = pydate(year(calc_date), month(calc_date), day(calc_date))

    result = er.py_obj.get_assumed_fill_iv(py_pf, Float64(s0), py_calc_date)
    return Dict{String, Float64}(pyconvert(String, pybuiltins.str(k)) => pyconvert(Float64, v) for (k, v) in result.items())
end

function get_assumed_fill_iv(er::Py, pf::Portfolio, s0, calc_date)
    py_pf = pf_to_py_pf(pf, calc_date)
    py_calc_date = pydate(year(calc_date), month(calc_date), day(calc_date))

    return er.get_assumed_fill_iv(py_pf, Float64(s0), py_calc_date)
end

"""
    get_iv_transition_matrices(er, arg) -> IVTransitionMatrices

Get IV transition matrices wrapped in a Julia struct.
"""
function get_iv_transition_matrices(er::PyEarningsReleaseSimulation, arg)
    m = er.py_obj.get_iv_transition_matrices(arg)
    return IVTransitionMatrices(
        m_dnlv01_worst_buy    = m.m_dnlv01_worst_buy,
        m_dnlv01_worst_sell   = m.m_dnlv01_worst_sell,
        m_dnlv01_best_buy     = m.m_dnlv01_best_buy,
        m_dnlv01_best_sell    = m.m_dnlv01_best_sell,
        m_dnlv01_estimated_buy  = m.m_dnlv01_estimated_buy,
        m_dnlv01_estimated_sell = m.m_dnlv01_estimated_sell,
        v_delta0_mid          = m.v_delta0_mid,
    )
end

"""
    solve(er) -> SolverResult

Run the portfolio optimizer and return a SolverResult.
"""
function solve(er::PyEarningsReleaseSimulation)
    res = er.py_obj.solve()
    return SolverResult(
        er,
        res.pf,
        res.pyo_model,
        res.pyo_inst,
    )
end