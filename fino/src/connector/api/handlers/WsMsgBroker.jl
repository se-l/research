module WsMsgBroker

using ProtoBuf
using HTTP
using HTTP.WebSockets: send, receive

using ..WS

export handler

# ─────────────────────────────────────────────────────────────────────────────
# Conflating queue
#
# Per (ws, channel) slot:
#   • pending  – the latest un-processed message, or nothing
#   • running  – true while a worker task is executing handle_on_msg
#
# Invariant: at most one worker task is alive per slot at any time.
# A new message always overwrites the previous pending (conflation).
# ─────────────────────────────────────────────────────────────────────────────

mutable struct ConflatingQueue
    pending ::Union{Nothing, Any}   # latest unprocessed message
    running ::Bool
    lock    ::ReentrantLock
    ConflatingQueue() = new(nothing, false, ReentrantLock())
end

# Global registry: channel => ConflatingQueue
# Access must be guarded by _registry_lock.
const _queues        = Dict{Any, ConflatingQueue}()
const _registry_lock = ReentrantLock()

function _get_or_create_queue(channel)::ConflatingQueue
    lock(_registry_lock) do
        get!(() -> ConflatingQueue(), _queues, channel)
    end
end

function _remove_all_queues!()
    lock(_registry_lock) do
        empty!(_queues)
    end
end


"""
    enqueue_and_process(ws, msg, handler_fn)

Drop `msg` into the conflating queue for `msg.channel`.
If no worker is running for that channel, spawn one.
"""
function enqueue_and_process(ws, msg, handler_fn)
    q = _get_or_create_queue(msg.channel)

    should_spawn = lock(q.lock) do
        q.pending = msg          # overwrite — conflation happens here
        if q.running
            false                # worker will pick up the new pending itself
        else
            q.running = true
            true                 # we must spawn the worker
        end
    end

    should_spawn || return

    Threads.@spawn begin
        try
            while true
                local current_msg = lock(q.lock) do
                    m = q.pending
                    q.pending = nothing
                    m
                end

                current_msg === nothing && break  # nothing left to do

                try
                    handler_fn(ws, current_msg)
                catch e
                    @error "[WsMsgBroker] handler error" channel=msg.channel exception=(e, catch_backtrace())
                end

                # Check if another message arrived while we were working
                peek = lock(q.lock) do
                    q.pending
                end
                peek === nothing && break
            end
        finally
            lock(q.lock) do
                q.running = false
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────

function handler(ws)::Nothing
    @info "[WsMsgBroker] Handler started for client"
    local msg_count = 0

    for data in ws
        try
            msg_count += 1

            if isempty(data)
                @warn "[WsMsgBroker] Received empty data (attempt $msg_count)"
                continue
            end

            @debug "[WsMsgBroker] Received $(length(data)) bytes (message #$msg_count)"

            msg = parse_pb(data, MessagePb)
            @info "[WsMsgBroker] Decoded message" channel=msg.channel action=msg.action id=msg.id

            # Heartbeat: always handled immediately, never conflated
            if msg.channel == ChannelPb.HB
                handle_on_heartbeat(ws, msg)
                continue
            end

            # All other channels: conflated per channel
            handler_fn = if msg.channel == ChannelPb.TARGET_PORTFOLIO
                TargetPortfoliosHandler.handle_on_msg
            elseif msg.channel == ChannelPb.STRESS_TEST_DS
                StressTestDsHandler.handle_on_msg
            elseif msg.channel == ChannelPb.KALMAN_INIT
                KalmanHandler.handle_on_msg
            elseif msg.channel == ChannelPb.REQUEST_SSVI_CALIBRATION
                SSVICalibrationHandler.handle_on_msg
            elseif msg.channel == ChannelPb.REQUEST_PF_RISK_SCENARIOS
                PfRiskScenarioHandler.handle_on_msg
            else
                @warn "[WsMsgBroker] Unknown channel" channel=msg.channel
                let ch = msg.channel
                    (ws, _) -> send(ws, Vector{UInt8}(codeunits("Unknown channel: $ch")))
                end
            end

            enqueue_and_process(ws, msg, handler_fn)

        catch e
            @error "[WsMsgBroker] Error processing message" exception=(e, catch_backtrace())
            #break
        end
    end

    # Single-client assumption: clear all queues on disconnect
    _remove_all_queues!()
    @info "[WsMsgBroker] Handler exited after $msg_count messages"
    return nothing
end

end # module WsMsgBroker