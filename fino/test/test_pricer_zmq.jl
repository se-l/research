# test_pricer_zmq.jl
#
# Integration test for PricerZMQ — sends every command and validates the reply.
# Assumes the pricer is already running on tcp://127.0.0.1:5555.
#
# Usage:
#   julia test_pricer_zmq.jl
#   julia test_pricer_zmq.jl tcp://127.0.0.1:5556   # custom endpoint

using ZMQ
using Test

const ENDPOINT = length(ARGS) > 0 ? ARGS[1] : "tcp://127.0.0.1:5555"

# ─────────────────────────────────────────────────────────────────────────────
# Command constants (mirrors PricerZMQ)
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

# ─────────────────────────────────────────────────────────────────────────────
# Wire encoding — mirrors the protocol in PricerZMQ exactly
# ─────────────────────────────────────────────────────────────────────────────

function encode_request(;
cmd          ::UInt8,
is_call      ::Bool    = true,
S            ::Float64 = 150.0,
K            ::Float64 = 150.0,
T            ::Float64 = 0.25,
sigma        ::Float64 = 0.30,
price        ::Float64 = 0.0,
calc_time_utc::Int64   = Int64(Dates.datetime2unix(DateTime(2025, 12, 18, 12, 0, 0))),
ticker       ::String  = "FDX",
yield_curve  ::String  = "",
d_spot       ::Float64 = 0.0,
d_sigma      ::Float64 = 0.0,
d_time       ::Float64 = 0.0,
time_steps   ::Int32   = Int32(0),
space_steps  ::Int32   = Int32(0)
)::Vector{UInt8}

    buf = IOBuffer()
    write(buf, cmd)
    write(buf, is_call ? UInt8(0x01) : UInt8(0x00))
    write(buf, S)
    write(buf, K)
    write(buf, T)
    write(buf, sigma)
    write(buf, price)
    write(buf, calc_time_utc)

    ticker_bytes = Vector{UInt8}(ticker)
    write(buf, Int32(length(ticker_bytes)))
    write(buf, ticker_bytes)

    yc_bytes = Vector{UInt8}(yield_curve)
    write(buf, Int32(length(yc_bytes)))
    write(buf, yc_bytes)

    write(buf, d_spot)
    write(buf, d_sigma)
    write(buf, d_time)
    write(buf, time_steps)
    write(buf, space_steps)

    take!(buf)
end

# ─────────────────────────────────────────────────────────────────────────────
# Reply decoding
# ─────────────────────────────────────────────────────────────────────────────

function decode_reply(raw::Vector{UInt8})
    isempty(raw) && return (error=true, msg="empty reply", value=NaN32)

    if raw[1] == 0xFF
        msg = length(raw) > 1 ? String(raw[2:end]) : "(no message)"
        return (error=true, msg=msg, value=NaN32)
    end

    length(raw) < 4 && return (error=true, msg="short reply ($(length(raw)) bytes)", value=NaN32)

    v = reinterpret(Float32, raw[1:4])[1]
    return (error=false, msg="", value=v)
end

# ─────────────────────────────────────────────────────────────────────────────
# Single request/reply round-trip
# ─────────────────────────────────────────────────────────────────────────────

function send_recv(socket, payload::Vector{UInt8})
    send(socket, payload)
    raw = recv(socket, Vector{UInt8})
    decode_reply(raw)
end

# ─────────────────────────────────────────────────────────────────────────────
# Test parameters — ATM call, 3-month tenor, 30% vol
# ─────────────────────────────────────────────────────────────────────────────

using Dates

const BASE = (
is_call       = true,
S             = 150.0,
K             = 150.0,
T             = 0.25,
sigma         = 0.30,
price         = 8.5,      # plausible ATM call price for IV inversion
calc_time_utc = Int64(round(datetime2unix(DateTime(2025, 12, 18, 12, 0, 0)))),
ticker        = "FDX",
yield_curve   = "",
d_spot        = 0.0,
d_sigma       = 0.0,
d_time        = 0.0,
time_steps    = Int32(0),
space_steps   = Int32(0)
)

# ─────────────────────────────────────────────────────────────────────────────
# Sanity predicate helpers
# ─────────────────────────────────────────────────────────────────────────────

is_finite_f32(v::Float32)   = isfinite(v)
is_positive(v::Float32)     = isfinite(v) && v > 0f0
is_negative(v::Float32)     = isfinite(v) && v < 0f0
in_range(v, lo, hi)         = isfinite(v) && lo < v < hi

# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

ctx    = Context()
socket = Socket(ctx, REQ)
connect(socket, ENDPOINT)
@info "Connected to $ENDPOINT"

try
    @testset "PricerZMQ protocol" begin

        @testset "CMD_PRICE (0x00)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_PRICE))
            @test !r.error             #     "Server error: $(r.msg)"
            @test is_positive(r.value)   #   "Price should be positive, got $(r.value)"
            @info "PRICE = $(r.value)"
        end

        @testset "CMD_IV (0x01)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_IV))
            @test !r.error                   #    "Server error: $(r.msg)"
            @test in_range(r.value, 0f0, 5f0) #  "IV should be in (0, 500%), got $(r.value)"
            @info "IV = $(r.value)"
        end

        @testset "CMD_DELTA (0x02)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_DELTA))
            @test !r.error                      #   "Server error: $(r.msg)"
            @test in_range(r.value, 0f0, 1f0)   #  "Call delta should be in (0,1), got $(r.value)"
            @info "DELTA = $(r.value)"
        end

        @testset "CMD_VEGA (0x03)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_VEGA))
            @test !r.error             #"Server error: $(r.msg)"
            @test is_positive(r.value) #"Vega should be positive, got $(r.value)"
            @info "VEGA = $(r.value)"
        end

        @testset "CMD_THETA (0x04)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_THETA))
            @test !r.error             #"Server error: $(r.msg)"
            @test is_negative(r.value) #"Theta should be negative, got $(r.value)"
            @info "THETA = $(r.value)"
        end

        @testset "CMD_GAMMA (0x05)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_GAMMA))
            @test !r.error             #"Server error: $(r.msg)"
            @test is_positive(r.value) #"Gamma should be positive, got $(r.value)"
            @info "GAMMA = $(r.value)"
        end

        @testset "CMD_SPEED (0x06)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_SPEED))
            @test !r.error               #"Server error: $(r.msg)"
            @test is_finite_f32(r.value) #"Speed should be finite, got $(r.value)"
            @info "SPEED = $(r.value)"
        end

        @testset "CMD_GAMMA_DECAY (0x07)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_GAMMA_DECAY))
            @test !r.error               #"Server error: $(r.msg)"
            @test is_finite_f32(r.value) #"GammaDecay should be finite, got $(r.value)"
            @info "GAMMA_DECAY = $(r.value)"
        end

        @testset "CMD_GAMMA_VOL (0x08)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_GAMMA_VOL))
            @test !r.error              # "Server error: $(r.msg)"
            @test is_finite_f32(r.value) #"GammaVol should be finite, got $(r.value)"
            @info "GAMMA_VOL = $(r.value)"
        end

        @testset "CMD_THETA_DECAY (0x09)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_THETA_DECAY))
            @test !r.error               #"Server error: $(r.msg)"
            @test is_finite_f32(r.value) #"ThetaDecay should be finite, got $(r.value)"
            @info "THETA_DECAY = $(r.value)"
        end

        @testset "CMD_VEGA_DECAY (0x0A)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_VEGA_DECAY))
            @test !r.error               #"Server error: $(r.msg)"
            @test is_finite_f32(r.value) #"VegaDecay should be finite, got $(r.value)"
            @info "VEGA_DECAY = $(r.value)"
        end

        @testset "CMD_VANNA (0x0B)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_VANNA))
            @test !r.error               #"Server error: $(r.msg)"
            @test is_finite_f32(r.value) #"Vanna should be finite, got $(r.value)"
            @info "VANNA = $(r.value)"
        end

        @testset "CMD_VOLGA (0x0C)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_VOLGA))
            @test !r.error               #"Server error: $(r.msg)"
            @test is_finite_f32(r.value) #"Volga should be finite, got $(r.value)"
            @info "VOLGA = $(r.value)"
        end

        @testset "CMD_MNY_FWD (0x0D)" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_MNY_FWD))
            @test !r.error               #"Server error: $(r.msg)"
            @test is_positive(r.value)   #"Forward moneyness should be positive, got $(r.value)"
            # ATM → moneyness ≈ 1.0 (slight drift from rates/divs)
            @test in_range(r.value, 0.5f0, 2f0) #"Moneyness out of expected range: $(r.value)"
            @info "MNY_FWD = $(r.value)"
        end

        @testset "Put delta sign" begin
            r = send_recv(socket, encode_request(; BASE..., cmd=CMD_DELTA, is_call=false))
            @test !r.error                          #"Server error: $(r.msg)"
            @test in_range(r.value, -1f0, 0f0)     #"Put delta should be in (-1,0), got $(r.value)"
            @info "PUT DELTA = $(r.value)"
        end

        @testset "Error reply for unknown command" begin
            payload = encode_request(; BASE..., cmd=0xFF)
            r = send_recv(socket, payload)
            # Server returns NaN32 for unknown cmd — not an error frame, but NaN
            @test !r.error #"Unexpected error frame for unknown cmd: $(r.msg)"
            @test isnan(r.value) #"Expected NaN for unknown command, got $(r.value)"
            @info "UNKNOWN CMD reply = $(r.value)"
        end

        @testset "Price consistency: IV round-trip" begin
            # 1. price a call
            rp = send_recv(socket, encode_request(; BASE..., cmd=CMD_PRICE))
            @test !rp.error #"Price error: $(rp.msg)"
            computed_price = Float64(rp.value)

            # 2. invert that price back to IV
            ri = send_recv(socket, encode_request(; BASE..., cmd=CMD_IV,
                price=computed_price))
            @test !ri.error #"IV error: $(ri.msg)"

            @test abs(ri.value - BASE.sigma) < 0.02 #"""
#                IV round-trip mismatch:
#                  input sigma = $(BASE.sigma)
#                  price       = $computed_price
#                  recovered   = $(ri.value)
#            """
            @info "IV round-trip: σ_in=$(BASE.sigma)  price=$(computed_price)  σ_out=$(ri.value)"
        end

    end  # @testset

finally
    close(socket)
    close(ctx)
end