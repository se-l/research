# moneyness.jl

using Interpolations

const DISCOUNT_RATE_MARKET = 0.04  # match your shared.constants.DiscountRateMarket

# ============================================================================
# Year fraction Act/365F
# ============================================================================

"""
    yearfrac_act365(t0::DateTime, t1::Union{Date, DateTime}) -> Float64
    yearfrac_act365(t0::DateTime, t1::Vector{<:Union{Date, DateTime}}) -> Vector{Float64}

Act/365F year fraction from `t0` to `t1`.
"""
function yearfrac_act365(t0::DateTime, t1::Union{Date, DateTime})::Float64
    return (date_to_sod(t1) - t0).value / (1000 * 3600 * 24) / 365.0
end

function yearfrac_act365(t0::DateTime, t1::Vector{<:Union{Date, DateTime}})::Vector{Float64}
    return [yearfrac_act365(t0, d) for d in t1]
end

# ============================================================================
# Zero rate interpolation (PCHIP on r(t)*t, flat extrapolation)
# ============================================================================

"""
    interp_zero_rate(times, rates, t) -> Vector{Float64}

Monotone PCHIP interpolation of zero rates. Interpolates on y(t)=r(t)*t
to avoid oscillation. Flat extrapolation beyond endpoints.
"""
function interp_zero_rate(curve_times::Vector{Float64}, curve_rates::Vector{Float64}, t::Vector{S})::Vector{Float64} where {S<:AbstractFloat}
    if isempty(curve_times)
        return fill(DISCOUNT_RATE_MARKET, length(t))
    end

    # Sort and deduplicate
    order = sortperm(curve_times)
    times = curve_times[order]
    rates = curve_rates[order]

    if length(times) >= 2
        uniq = [true; times[2:end] .!= times[1:end-1]]
        times = times[uniq]
        rates = rates[uniq]
    end

    if length(times) == 1
        return fill(rates[1], length(t))
    end

    left_rate  = rates[1]
    right_rate = rates[end]

    out = Vector{Float64}(undef, length(t))

    ix_left  = t .<= times[1]
    ix_right = t .>= times[end]
    ix_mid   = .!(ix_left .| ix_right)

    out[ix_left]  .= left_rate
    out[ix_right] .= right_rate

    if any(ix_mid)
        tm = t[ix_mid]
        y  = rates .* times   # interpolate r(t)*t

        itp = interpolate(times, y, SteffenMonotonicInterpolation())
        ym  = itp.(tm)

        out_mid = Vector{Float64}(undef, length(tm))
        ix_zero = tm .== 0.0
        out_mid[ix_zero]  .= left_rate
        out_mid[.!ix_zero] = ym[.!ix_zero] ./ tm[.!ix_zero]
        out[ix_mid] = out_mid
    end

    return out
end

# Convenience wrapper accepting a curve object with .times and .rates
function interp_zero_rate(curve, t::Vector{S})::Vector{Float64} where {S<:AbstractFloat}
    return interp_zero_rate(Float64.(curve.times), Float64.(curve.rates), t)
end

# ============================================================================
# Discount factor from semi-annual zero rates
# ============================================================================

"""
    discount_factor_from_zero(curve, t) -> Vector{Float64}

P(0,t) = (1 + z(t)/2)^(-2t), semi-annual compounding.
"""
function discount_factor_from_zero(curve, t::Vector{S})::Vector{Float64} where {S<:AbstractFloat}
    z = interp_zero_rate(curve, t)
    return (1.0 .+ z ./ 2.0) .^ (-2.0 .* t)
end

# ============================================================================
# Add year fraction to a DateTime
# ============================================================================

"""
    add_year_fraction(t0::DateTime, tenor::Float64) -> DateTime
    add_year_fraction(t0::DateTime, tenor::Vector{Float64}) -> Vector{DateTime}
"""
function add_year_fraction(t0::DateTime, tenor::S)::DateTime where {S<:AbstractFloat}
    return t0 + Millisecond(round(Int, tenor * 365 * 24 * 3600 * 1000))
end

function add_year_fraction(t0::DateTime, tenor::Vector{S})::Vector{DateTime} where {S<:AbstractFloat}
    return [add_year_fraction(t0, τ) for τ in tenor]
end

# ============================================================================
# Forward price from curve + discrete dividends / continuous yield
# ============================================================================

"""
    forward_from_curve_and_divs(spot, t0, t1, curve; cash_dividends, dividend_yield) -> Vector{Float64}

F(0,T) with discrete cash dividends or continuous dividend yield q.
"""
function forward_from_curve_and_divs(
    spot::Vector{S},
    t0::DateTime,
    t1::Vector{DateTime},
    curve;
    cash_dividends = nothing
)::Vector{Float64} where {S<:AbstractFloat}
    T   = yearfrac_act365(t0, t1)
    P0T = discount_factor_from_zero(curve, T)

    if !isnothing(cash_dividends) && !isempty(cash_dividends)
        pv_div = zeros(Float64, length(spot))
        t0_date = Date(t0)
        t1_dates = Date.(t1)

        for d in cash_dividends
            exd = d.ex_date
            exd > t0_date || continue          # ex-date must be after t0
            mask = exd .<= t1_dates
            any(mask) || continue

            t_div  = yearfrac_act365(t0, DateTime(exd))
            P0div  = discount_factor_from_zero(curve, [t_div])[1]
            pv_div[mask] .+= Float64(d.amount) * P0div
        end

        return (spot .- pv_div) ./ P0T
    end

    # No dividends: pure yield curve discounting F = S / P(0,T)
    return spot ./ P0T
end

# ============================================================================
# Forward moneyness  K / F(0,T)
# ============================================================================

"""
    get_moneyness_fwd(equity, strike, spot, tenor, t0; yield_curve_in) -> Vector{Float64}

K / F(0,T)  where F is forward from curve + dividends.
`tenor` is in years. `t0` must be a `DateTime`.
"""
function get_moneyness_fwd(
    equity,
    strike::Vector{S},
    spot::Vector{T},
    tenor::Vector{U},
    t0::DateTime;
    yield_curve_in = nothing,
)::Vector{Float64} where {S<:AbstractFloat, T<:AbstractFloat, U<:AbstractFloat}
    yield_curve = isnothing(yield_curve_in) ? get_last_zero_curve(Date(t0), equity) : yield_curve_in

    t1 = add_year_fraction(t0, tenor)
    cash_dividends = get_dividends(uppercase(equity.symbol), t0)

    spot_fwd = forward_from_curve_and_divs(
        spot, t0, t1, yield_curve;
        cash_dividends = cash_dividends,
    )

    return strike ./ spot_fwd
end

# ============================================================================
# Log forward moneyness  log(K / F(0,T))
# ============================================================================

"""
    get_moneyness_fwd_ln(equity, strike, spot, tenor, t0; yield_curve_in) -> Vector{Float64}

k = log(K / F(0,T))
"""
function get_moneyness_fwd_ln(
    equity,
    strike::Vector{S},
    spot::Vector{T},
    tenor::Vector{U},
    t0::DateTime;
    yield_curve_in = nothing,
)::Vector{Float64} where {S<:AbstractFloat, T<:AbstractFloat, U<:AbstractFloat}
    return log.(get_moneyness_fwd(equity, strike, spot, tenor, t0; yield_curve_in))
end