# using PyCall   # ← remove top-level import
# using DataFrames

# ============================================================================
# Abstract interface — allows swapping to a pure Julia impl later
# ============================================================================

abstract type AbstractEarningsIVDropRegressor end

"""
    predict(regressor, moneyness_fwd_ln, tenor; kwargs...) -> Vector{Float64}

Predict IV drop from moneyness and tenor inputs.
"""
function predict end

"""
    load_model!(regressor, path)
    load_model(T, path) -> regressor

Load a serialized model from disk.
"""
function load_model! end

# ============================================================================
# Python wrapper via PyCall (lazy-loaded)
# ============================================================================

"""
    PyEarningsIVDropPolyRegressorV3

Wraps the Python `EarningsIVDropPolyRegressorV3` class via PyCall.
Holds a reference to the Python object; all predict calls delegate to it.
"""
mutable struct PyEarningsIVDropPolyRegressorV3 <: AbstractEarningsIVDropRegressor
    py_model::Any  # PyObject, but typed as Any to avoid requiring PyCall at parse time
end

# Lazy import — avoids importing PyCall/sklearn/joblib at module load time
const _py_mod = Ref{Any}()

function _get_py_mod()
    if !isassigned(_py_mod)
        @eval begin
            Base.require(Base.PkgId(Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0"), "PyCall"))
        end
        _py_mod[] = PyCall.pyimport("options.volatility.estimators.earnings_iv_drop_poly_regressor")
    end
    return _py_mod[]
end

"""
    PyEarningsIVDropPolyRegressorV3(; model_nm=nothing)

Construct by calling the Python constructor.
"""
function PyEarningsIVDropPolyRegressorV3(; model_nm::Union{String, Nothing}=nothing)
    py_cls = _get_py_mod().EarningsIVDropPolyRegressorV3
    py_obj = if model_nm === nothing
        py_cls()
    else
        py_cls(model_nm=model_nm)
    end
    return PyEarningsIVDropPolyRegressorV3(py_obj)
end

"""
    predict(reg::PyEarningsIVDropPolyRegressorV3, moneyness_fwd_ln, tenor;
            min_moneyness=-99.0, max_moneyness=99.0, min_tenor=0.0) -> Vector{Float64}

Delegate to Python `predict()`, accepting Julia vectors and returning a Julia `Vector{Float64}`.
"""
function predict(
    reg::PyEarningsIVDropPolyRegressorV3,
    moneyness_fwd_ln::AbstractVector{<:Real},
    tenor::AbstractVector{<:Real};
    min_moneyness::Float64 = -99.0,
    max_moneyness::Float64 = 99.0,
    min_tenor::Float64 = 0.0,
)::Vector{Float64}
    pd = PyCall.pyimport("pandas")
    df = pd.DataFrame(Dict(
        "moneyness_fwd_ln" => moneyness_fwd_ln,
        "tenor" => tenor,
    ))
    result = reg.py_model.predict(df, min_moneyness=min_moneyness, max_moneyness=max_moneyness, min_tenor=min_tenor)
    return collect(Float64, result)
end

"""
    load_model!(reg::PyEarningsIVDropPolyRegressorV3, path::String)

Reload the underlying sklearn pipeline from a `.joblib` file.
"""
function load_model!(reg::PyEarningsIVDropPolyRegressorV3, path::String)
    reg.py_model.load_model(path)
    return reg
end

function load_model!(::Type{PyEarningsIVDropPolyRegressorV3}, path::String)
    reg = PyEarningsIVDropPolyRegressorV3()  # dummy init, will be overwritten
    reg.py_model.load_model(path)
    return reg
end