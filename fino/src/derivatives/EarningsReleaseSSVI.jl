module EarningsReleaseSSVI

#using ..PricingEngine
import Fino.PricingEngine: get_v_iv_fd, get_v_price_fd

# ────────────────────────────────────────────────────────────────────────────
# Vectorized NLV delta matrix
# Replaces the Python double-loop in get_nlv_delta_matrix()
#
# Inputs (all Float32 vectors, length = n_options):
#   spots_0, strikes, tenors_0, v_is_call     — option contract data at t0
#   prices_bid_0, prices_ask_0                — market bid/ask prices at t0
#
# For each ds scenario (length = n_ds):
#   spots_1[i] = s0 * ds_ret[i]              — shifted spot
#   tenors_1                                  — tenors at t1 (may differ from t0)
#   ivs_1[i, :]                               — SSVI model IV for scenario i (n_ds × n_options)
#
# Returns four matrices (n_ds × n_options):
#   dnlv_worst_long, dnlv_worst_short, dnlv_best_long, dnlv_best_short
# ────────────────────────────────────────────────────────────────────────────
function get_nlv_delta_matrix(
    # t0 data
    prices_bid_0    :: Vector{Float32},    # market bid prices at t0
    prices_ask_0    :: Vector{Float32},    # market ask prices at t0
    spots_0         :: Vector{Float32},    # spot at t0 (same value repeated n_options times)
    strikes         :: Vector{Float32},
    tenors_0        :: Vector{Float32},    # time-to-expiry at t0 in years
    v_is_call       :: Vector{UInt8},
    rates_curve     :: Vector{Float32},
    rates_times     :: Vector{Float32},
    div_amounts     :: Vector{Float32},
    div_times       :: Vector{Float32},
    multiplier      :: Float32,            # e.g. 100f0 for equity options
    # t1 scenario data
    v_spots_1       :: Vector{Float32},    # one spot per ds scenario  (n_ds,)
    tenors_1        :: Vector{Float32},    # tenors at t1              (n_options,)
    m_ivs_1         :: Matrix{Float32};    # SSVI model IVs            (n_ds × n_options)
    time_steps :: Int = 200,
    space_steps:: Int = 200,
)
    n_options = length(strikes)
    n_ds      = length(v_spots_1)

    # Guard: nothing to price
    if n_options == 0 || n_ds == 0
        empty = Matrix{Float32}(undef, n_ds, n_options)
        return Vector{Float32}(undef, n_options), Vector{Float32}(undef, n_options),
               empty, empty, empty, empty
    end

    # ── Validate inputs before any GPU call ────────────────────────────────
    @info "get_nlv_delta_matrix inputs" n_options n_ds time_steps space_steps
#    @info "  tenors_0" tenors_0 min=minimum(tenors_0) max=maximum(tenors_0) any_zero=any(iszero, tenors_0) any_neg=any(<(0), tenors_0)
#    @info "  tenors_1" tenors_1 min=minimum(tenors_1) max=maximum(tenors_1) any_zero=any(iszero, tenors_1) any_neg=any(<(0), tenors_1)
#    @info "  spots_0"  min=minimum(spots_0)  max=maximum(spots_0)
#    @info "  strikes"  min=minimum(strikes)  max=maximum(strikes)
#    @info "  prices_bid_0" min=minimum(prices_bid_0) max=maximum(prices_bid_0) any_neg=any(<(0), prices_bid_0)
#    @info "  prices_ask_0" min=minimum(prices_ask_0) max=maximum(prices_ask_0) any_neg=any(<(0), prices_ask_0)
#    @info "  v_spots_1" min=minimum(v_spots_1) max=maximum(v_spots_1)

    bad_t0 = findall(t -> t <= 0, tenors_0)
    if !isempty(bad_t0)
        @error "Zero/negative tenors_0 at indices" bad_t0 values=tenors_0[bad_t0] strikes=strikes[bad_t0]
        error("DivideError will occur: tenors_0 has $(length(bad_t0)) zero/negative value(s)")
    end

    bad_t1 = findall(t -> t <= 0, tenors_1)
    if !isempty(bad_t1)
        @error "Zero/negative tenors_1 at indices" bad_t1 values=tenors_1[bad_t1] strikes=strikes[bad_t1]
        error("DivideError will occur: tenors_1 has $(length(bad_t1)) zero/negative value(s)")
    end

    # ── Step 1: NLV at t0 directly from market bid/ask prices ─────────────
    nlv0_bid = prices_bid_0 .* multiplier
    nlv0_ask = prices_ask_0 .* multiplier

    # ── Step 2: price under each dS scenario at t1 ────────────────────────
    # Flatten all scenarios into one big GPU batch: (n_ds * n_options,)
    n_total    = n_ds * n_options

    spots_1_flat  = repeat(v_spots_1,  inner = n_options)   # [s1, s1, ..., s2, s2, ...]
    strikes_flat  = repeat(strikes,    outer = n_ds)
    tenors_1_flat = repeat(tenors_1,   outer = n_ds)
    is_call_flat  = repeat(v_is_call,  outer = n_ds)
    ivs_1_flat    = collect(vec(m_ivs_1'))                  # row-major → flat

    prices_1_flat = get_v_price_fd(
        spots_1_flat, strikes_flat, tenors_1_flat,
        ivs_1_flat, is_call_flat,
        rates_curve, rates_times, div_amounts, div_times;
            time_steps, space_steps,
    ) .* multiplier

    # Reshape back to (n_ds × n_options)
    nlv1 = reshape(prices_1_flat, n_options, n_ds)'   # (n_ds × n_options)

    # ── Step 3: compute the four dNLV matrices ─────────────────────────────
    nlv0_bid_row = reshape(nlv0_bid, 1, n_options)    # broadcast over n_ds
    nlv0_ask_row = reshape(nlv0_ask, 1, n_options)

    dnlv_worst_long  = nlv1 .- nlv0_ask_row   # buy at ask, sell at model
    dnlv_worst_short = nlv1 .- nlv0_bid_row   # sell at bid, buy back at model
    dnlv_best_long   = nlv1 .- nlv0_bid_row
    dnlv_best_short  = nlv1 .- nlv0_ask_row

    return nlv0_bid, nlv0_ask, dnlv_worst_long, dnlv_worst_short, dnlv_best_long, dnlv_best_short
end

export get_nlv_delta_matrix

end # module EarningsReleaseSSVI