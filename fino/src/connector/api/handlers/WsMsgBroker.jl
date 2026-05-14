module WsMsgBroker

using ProtoBuf
using HTTP
using HTTP.WebSockets: send, receive

using ..WS

export handler

"""
    handler(ws::WebSocket) -> Nothing

WebSocket message broker handler that routes incoming messages
to appropriate handlers based on the channel type.
"""
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

            @info "[WsMsgBroker] Received $(length(data)) bytes (message #$msg_count)"

            msg = parse_pb(data, MessagePb)
            @info "[WsMsgBroker] Decoded message" channel=msg.channel action=msg.action id=msg.id

            # Route based on channel type
            if msg.channel == ChannelPb.HB
                @info "[WsMsgBroker] Routing to handle_on_heartbeat"
                handle_on_heartbeat(ws, msg)
            elseif msg.channel == ChannelPb.TARGET_PORTFOLIO
                @info "[WsMsgBroker] Routing to TargetPortfolios"
                TargetPortfoliosHandler.handle_on_msg(ws, msg)
            elseif msg.channel == ChannelPb.STRESS_TEST_DS
                @info "[WsMsgBroker] Routing to StressTestDsHandler"
                StressTestDsHandler.handle_on_msg(ws, msg)
            elseif msg.channel == ChannelPb.KALMAN_INIT
                @info "[WsMsgBroker] Routing to Kalman"
                KalmanHandler.handle_on_msg(ws, msg)
            elseif msg.channel == ChannelPb.REQUEST_SSVI_CALIBRATION
                @info "[WsMsgBroker] Routing to SSVICalibrationHandler"
                SSVICalibrationHandler.handle_on_msg(ws, msg)
            elseif msg.channel == ChannelPb.REQUEST_PF_RISK_SCENARIOS
                @info "[WsMsgBroker] Routing to PfRiskScenarioHandler"
                PfRiskScenarioHandler.handle_on_msg(ws, msg)
            else
                @warn "[WsMsgBroker] Unknown channel" channel=msg.channel
                error_msg = "Unknown channel: $(msg.channel)"
                send(ws, Vector{UInt8}(codeunits(error_msg)))
            end
        catch e
            @error "[WsMsgBroker] Error processing message" exception=(e, catch_backtrace())
            break
        end
    end

    @info "[WsMsgBroker] Handler exited after $msg_count messages"
    return nothing
end

end # module WsMsgBroker