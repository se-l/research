module PricerZMQ

#=
Description: ZMQ REP server wrapping PricingEngine greek calculations.
Author: seb
Date: 4/11/2026
=#

export start_pricer

using ZMQ
using Dates
using ..PricingEngine
using ..DividendManager
using ..YieldCurve

# ─────────────────────────────────────────────────────────────────────────────
# Command constants
# ─────────────────────────────────────────────────────────────────────────────

const CMD_PRICE       = 0x00
const CMD_IV          = 0x01
const CMD_DELTA       = 0x02
const CMD_VEGA        = 0x03
const CMD_THETA       = 0x04
const CMD_GAMMA       = 0x05
const CMD_SPEED       = 0x06
const CMD_GAMMA_DECAY = 0x07
const CMD_GAMMA_VOL   = 0x08
const CMD_THETA_DECAY = 0x09
const CMD_VEGA_DECAY  = 0x0A
const CMD_VANNA       = 0x0B
const CMD_VOLGA       = 0x0C
const CMD_MNY_FWD     = 0x0D

const CMD_NAMES = Dict{UInt8,String}(
    CMD_PRICE       => "PRICE",
    CMD_IV          => "IV",
    CMD_DELTA       => "DELTA",
    CMD_VEGA        => "VEGA",
    CMD_THETA       => "THETA",
    CMD_GAMMA       => "GAMMA",
    CMD_SPEED       => "SPEED",
    CMD_GAMMA_DECAY => "GAMMA_DECAY",
    CMD_GAMMA_VOL   => "GAMMA_VOL",
    CMD_THETA_DECAY => "THETA_DECAY",
    CMD_VEGA_DECAY  => "VEGA_DECAY",
    CMD_VANNA       => "VANNA",
    CMD_VOLGA       => "VOLGA",
    CMD_MNY_FWD     => "MNY_FWD",
)

# ─────────────────────────────────────────────────────────────────────────────
# Wire protocol
# ─────────────────────────────────────────────────────────────────────────────
#
# All multi-byte integers/floats are little-endian.
#
#   1  byte   cmd
#   1  byte   is_call        (0x01 = call, 0x00 = put)
#   8  bytes  S              Float64
#   8  bytes  K              Float64
#   8  bytes  T              Float64  (years to expiry)
#   8  bytes  sigma          Float64  (ignored for CMD_IV)
#   8  bytes  price          Float64  (market price, used only by CMD_IV)
#   8  bytes  calc_time_utc  Int64    (Unix timestamp seconds, UTC)
#   4  bytes  len_ticker     Int32
#   …         ticker         UTF-8 bytes
#   4  bytes  len_yc         Int32
#   …         yield_curve    UTF-8 bytes  (e.g. "YC")
#   8  bytes  d_spot         Float64  (0.0 → default 0.1)
#   8  bytes  d_sigma        Float64  (0.0 → default 0.001)
#   8  bytes  d_time         Float64  (0.0 → default 1/365)
#   4  bytes  time_steps     Int32    (0 → default 200)
#   4  bytes  space_steps    Int32    (0 → default 200)
#
# Reply (success):  4 bytes Float32 LE
# Reply (error):    0xFF byte + UTF-8 error message

# ─────────────────────────────────────────────────────────────────────────────
# Request — as parsed directly from the wire (no market data yet)
# ─────────────────────────────────────────────────────────────────────────────

struct Request
    cmd           ::UInt8
    is_call       ::Bool
    S             ::Float32
    K             ::Float32
    T             ::Float32
    sigma         ::Float32
    price         ::Float32   # market price (IV only)
    calc_time_utc ::Int64     # Unix timestamp
    ticker        ::String
    yield_curve   ::String
    d_spot        ::Float32
    d_sigma       ::Float32
    d_time        ::Float32
    time_steps    ::Int32
    space_steps   ::Int32
end

# ─────────────────────────────────────────────────────────────────────────────
# ResolvedRequest — after market data has been fetched
# ─────────────────────────────────────────────────────────────────────────────

struct ResolvedRequest
    cmd         ::UInt8
    is_call     ::Bool
    S           ::Float32
    K           ::Float32
    T           ::Float32
    sigma       ::Float32
    price       ::Float32
    rates_times ::Vector{Float32}
    rates_curve ::Vector{Float32}
    div_times   ::Vector{Float32}
    div_amounts ::Vector{Float32}
    d_spot      ::Float32
    d_sigma     ::Float32
    d_time      ::Float32
    time_steps  ::Int32
    space_steps ::Int32
end

# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

function parse_request(buf::Vector{UInt8})::Request
    pos = Ref(1)

    function read_byte()
        v = buf[pos[]]
        pos[] += 1
        v
    end
    function read_f64()
        v = reinterpret(Float64, buf[pos[]:pos[]+7])[1]
        pos[] += 8
        Float32(v)
    end
    function read_i32()
        v = reinterpret(Int32, buf[pos[]:pos[]+3])[1]
        pos[] += 4
        Int(v)
    end
    function read_i64()
        v = reinterpret(Int64, buf[pos[]:pos[]+7])[1]
        pos[] += 8
        v
    end
    function read_string()
        n = read_i32()
        s = String(buf[pos[]:pos[]+n-1])
        pos[] += n
        s
    end

    cmd           = read_byte()
    is_call       = read_byte() == 0x01
    S             = read_f64()
    K             = read_f64()
    T             = read_f64()
    sigma         = read_f64()
    price         = read_f64()
    calc_time_utc = read_i64()
    ticker        = read_string()
    yield_curve   = read_string()

    d_spot_raw  = read_f64()
    d_sigma_raw = read_f64()
    d_time_raw  = read_f64()
    ts_raw      = read_i32()
    ss_raw      = read_i32()

    d_spot  = d_spot_raw  > 0f0 ? d_spot_raw  : 1f-1
    d_sigma = d_sigma_raw > 0f0 ? d_sigma_raw : 1f-3
    d_time  = d_time_raw  > 0f0 ? d_time_raw  : 1f0 / 365f0
    time_steps  = ts_raw > 0 ? ts_raw : Int32(200)
    space_steps = ss_raw > 0 ? ss_raw : Int32(200)

    Request(cmd, is_call, S, K, T, sigma, price,
            calc_time_utc, ticker, yield_curve,
            d_spot, d_sigma, d_time,
            time_steps, space_steps)
end

# ─────────────────────────────────────────────────────────────────────────────
# Market data enrichment
# Fetches dividend schedule and zero curve, returns a ResolvedRequest.
# ─────────────────────────────────────────────────────────────────────────────

function unix_to_date(ts::Int64)::Date
    # Dates.unix2datetime returns a DateTime in UTC
    Date(Dates.unix2datetime(ts))
end

function enrich_request(r::Request)::ResolvedRequest
    calc_date = unix_to_date(r.calc_time_utc)

    # Dividends: returns (Vector{Float}, Vector{Float}) = (amounts, times)
    div_amounts, div_times = get_dividend_amount_times(r.ticker, calc_date)

    # Zero curve: returns object with .times and .rates

    zc = YieldCurve.get_last_zero_curve(calc_date, r.ticker, -14)
    # Log

    ResolvedRequest(
        r.cmd, r.is_call, r.S, r.K, r.T, r.sigma, r.price,
        zc.times, zc.rates,
        div_times, div_amounts,
        r.d_spot, r.d_sigma, r.d_time,
        r.time_steps, r.space_steps)
end


# ─────────────────────────────────────────────────────────────────────────────
# Forward moneyness:  K / F(0,T)
#
# F(0,T) = (S - Σ PV(Dᵢ)) / P(0,T)
# P(0,T) = exp(-Z(T) * T)   where Z(T) is linearly interpolated zero rate
# PV(Dᵢ) = Dᵢ * exp(-Z(tᵢ) * tᵢ)   for ex-dates tᵢ ∈ (0, T]
# ─────────────────────────────────────────────────────────────────────────────

function _interp_zero(rates_times::Vector{Float32}, rates_curve::Vector{Float32}, t::Float32)::Float32
    isempty(rates_times) && return 0f0
    t <= rates_times[1]   && return rates_curve[1]
    t >= rates_times[end] && return rates_curve[end]
    for i in 1:length(rates_times)-1
        if rates_times[i] <= t <= rates_times[i+1]
            t1, t2 = rates_times[i], rates_times[i+1]
            r1, r2 = rates_curve[i], rates_curve[i+1]
            return r1 + (r2 - r1) * (t - t1) / (t2 - t1)
        end
    end
    return rates_curve[end]
end

@inline function _discount(rates_times::Vector{Float32}, rates_curve::Vector{Float32}, t::Float32)::Float32
    z = _interp_zero(rates_times, rates_curve, t)
    exp(-z * t)
end

function get_mny_fwd(
    S            ::Float32,
    K            ::Float32,
    T            ::Float32,
    rates_times  ::Vector{Float32},
    rates_curve  ::Vector{Float32},
    div_amounts  ::Vector{Float32},
    div_times    ::Vector{Float32}   # time-to-ex-date in years, already > 0
)::Float32
    (T <= 0f0 || S <= 0f0 || K <= 0f0) && return NaN32

    # P(0,T)
    P0T = _discount(rates_times, rates_curve, T)
    P0T <= 0f0 && return NaN32

    # Σ PV(Dᵢ) for ex-dates in (0, T]
    pv_divs = 0f0
    for i in eachindex(div_amounts)
        t_div = div_times[i]
        0f0 < t_div <= T || continue
        pv_divs += div_amounts[i] * _discount(rates_times, rates_curve, t_div)
    end

    fwd = (S - pv_divs) / P0T
    fwd <= 0f0 && return NaN32

    return K / fwd
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-command dispatch
# ─────────────────────────────────────────────────────────────────────────────

function dispatch(r::ResolvedRequest)::Float32
    rc = r.rates_curve
    rt = r.rates_times
    da = r.div_amounts
    dt = r.div_times
    ts = r.time_steps
    ss = r.space_steps

    if r.cmd == CMD_PRICE
        get_price_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_IV
        get_iv_fd(
            r.price, r.S, r.K, r.T, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_DELTA
        get_delta_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_spot=r.d_spot, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_VEGA
        get_vega_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_sigma=r.d_sigma, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_THETA
        get_theta_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_time=r.d_time, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_GAMMA
        get_gamma_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_spot=r.d_spot, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_SPEED
        get_speed_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_spot=r.d_spot, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_GAMMA_DECAY
        get_gamma_decay_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_spot=r.d_spot, d_time=r.d_time, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_GAMMA_VOL
        get_gamma_vol_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_spot=r.d_spot, d_sigma=r.d_sigma, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_THETA_DECAY
        get_theta_decay_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_time=r.d_time, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_VEGA_DECAY
        get_vega_decay_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_sigma=r.d_sigma, d_time=r.d_time, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_VANNA
        get_vanna_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_spot=r.d_spot, d_sigma=r.d_sigma, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_VOLGA
        get_volga_fd(
            r.S, r.K, r.T, r.sigma, r.is_call ? 0x01 : 0x00,
            rc, rt, da, dt;
            d_sigma=r.d_sigma, time_steps=ts, space_steps=ss)

    elseif r.cmd == CMD_MNY_FWD
        get_mny_fwd(r.S, r.K, r.T,
                        r.rates_times, r.rates_curve,
                        r.div_amounts, r.div_times)

    else
        NaN32
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Reply helpers
# ─────────────────────────────────────────────────────────────────────────────

function reply_f32(socket, v::Float32)
    payload = vcat(UInt8[0x02], reinterpret(UInt8, [v]))
    send(socket, payload)
#    send(socket, copy(reinterpret(UInt8, [v])))
end

"""
    reply_error(socket, msg::String, stacktrace)

Serialise a `0x03` error frame followed by `msg \\n\\n--- stack ---\\n stacktrace`
as raw UTF-8 bytes.  The caller must pass `catch_backtrace()` as `stacktrace`.
"""
function reply_error(socket, msg::String, raw_stack::Vector)
    full = if isempty(raw_stack)
        msg
    else
        # stacktrace() (lowercase) resolves Ptr/InterpreterIP → StackFrame[]
        resolved = stacktrace(raw_stack)
        string(
            msg,
            "\n\n--- Stack ---\n",
            sprint(io -> show(io, MIME"text/plain"(), resolved))
        )
    end
    @error "PricerZMQ error" full
    payload = vcat(UInt8[0x03], Vector{UInt8}(full))
    send(socket, payload)
end

reply_error(socket, msg::String) = reply_error(socket, msg, [])

# ─────────────────────────────────────────────────────────────────────────────
# Server loop
# ─────────────────────────────────────────────────────────────────────────────

function start_pricer(
    protocol = get(ENV, "PricerProtocol", "tcp"),
    host = get(ENV, "PricerHost", "0.0.0.0"),
    port::Int = parse(Int, get(ENV, "PricerPort", "8102"))
)
    if protocol == "ipc"
        start_pricer_at("ipc://$host.$port")
    end
    start_pricer_at("tcp://$(host):$port")
end


function start_pricer_at(endpoint::String)
    @info "PricerZMQ.start_pricer_at(): $endpoint"

    # ─────────────────────────────────────────────────────────────────────────────
     # JIT Compilation Warm-up
     # The first call to any function in Julia triggers JIT compilation, which can
     # add significant latency. We perform a set of dummy calculations here at
     # startup to pre-compile the most common code paths, including the CUDA kernels.
     # ─────────────────────────────────────────────────────────────────────────────
#     @info "Performing JIT compilation warm-up..."
#     try
#         # 1. Warm up the scalar CPU price path (get_price_fd)
#         dummy_req_price = ResolvedRequest(
#             CMD_PRICE,      # cmd
#             true,           # is_call
#             100f0,          # S
#             100f0,          # K
#             1f0,            # T
#             0.2f0,          # sigma
#             0f0,            # price (not used for CMD_PRICE)
#             [0.5f0, 1f0],   # rates_times
#             [0.05f0, 0.05f0], # rates_curve
#             [0.5f0],        # div_times
#             [1f0],          # div_amounts
#             0.1f0,          # d_spot
#             0.001f0,        # d_sigma
#             1/365f0,        # d_time
#             200,            # time_steps
#             200             # space_steps
#         )
#         p_res = dispatch(dummy_req_price)
#
#         # 2. Warm up a vectorized greek path (e.g., get_vega_fd)
#         # This is crucial for compiling the GPU kernels ahead of time.
#         dummy_req_vega = ResolvedRequest(
#             CMD_VEGA,       # cmd
#             true,           # is_call
#             100f0, 100f0, 1f0, 0.2f0, 0f0, # S, K, T, sigma, price
#             [0.5f0, 1f0], [0.05f0, 0.05f0], # rates
#             [0.5f0], [1f0],                 # divs
#             0.1f0, 0.001f0, 1/365f0,        # deltas
#             200, 200                        # steps
#         )
#         v_res = dispatch(dummy_req_vega)
#         @info "Warm-up complete. (Dummy results: Price=$p_res, Vega=$v_res)"
#     catch ex
#         @error "Warm-up failed, first request may be slow." sprint(showerror, ex)
#     end

    ctx    = Context()
    socket = Socket(ctx, REP)
    bind(socket, endpoint)
    @info "PricerZMQ listening on $endpoint"

    try
        cmd_counts = Dict{UInt8,Int}()
        while true
            buf = recv(socket, Vector{UInt8})
            try
                local req      = parse_request(buf)
                local resolved = enrich_request(req)
                local result   = dispatch(resolved)
                cmd_name = get(CMD_NAMES, req.cmd, "UNKNOWN(0x$(string(req.cmd, base=16)))")
                cmd_counts[req.cmd] = get(cmd_counts, req.cmd, 0) + 1
                if isnan(result)
                    @warn "[$(cmd_counts[req.cmd])] cmd=$cmd_name returned NaN" req resolved
                else
                     @debug "[$(cmd_counts[req.cmd])] cmd=$cmd_name = $result"
                end
                if cmd_counts[req.cmd] % 1000 == 0
                    @info "[$(cmd_counts[req.cmd])] cmd=$cmd_name = $result"
                end
                reply_f32(socket, result)
            catch ex
                reply_error(socket, sprint(showerror, ex), catch_backtrace())
            end
        end
    finally
        close(socket)
        close(ctx)
    end
end

end # module PricerZMQ