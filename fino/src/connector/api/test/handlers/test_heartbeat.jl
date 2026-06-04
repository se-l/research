# test/test_heartbeat.jl

using Test
using HTTP
using HTTP.WebSockets: WebSocket
using ProtoBuf
using Fino.WS: ChannelPb, ActionPb, MessagePb

const WS_URL = "ws://127.0.0.1:8002/ws"

"""Encode a MessagePb to protobuf bytes."""
function encode_msg(msg::MessagePb)::Vector{UInt8}
    buf = IOBuffer()
    encoder = ProtoBuf.ProtoEncoder(buf)
    ProtoBuf.encode(encoder, msg)
    return take!(buf)
end

"""Decode protobuf bytes into a MessagePb."""
function decode_msg(data::Vector{UInt8})::MessagePb
    decoder = ProtoBuf.ProtoDecoder(IOBuffer(data))
    return ProtoBuf.decode(decoder, MessagePb)
end

@async begin
    @info "[CLIENT] Connecting to ws://127.0.0.1:8002/ws ..."
    HTTP.WebSockets.open("ws://127.0.0.1:8002/ws") do ws
        @info "[CLIENT] Connected"

        sub_msg = MessagePb(ChannelPb.HB, "debug-hb-001", ActionPb.SUBSCRIBE, UInt8[])
        bytes = encode_msg(sub_msg)
        @info "[CLIENT] Sending SUBSCRIBE ($(length(bytes)) bytes)"
        WebSockets.send(ws, bytes)

        for i in 1:5
            data = WebSockets.receive(ws)
            if !isempty(data)
                @info "[CLIENT] Received $(length(data)) bytes"
                try
                    resp = decode_msg(data)
                    @info "[CLIENT] Decoded response" channel=resp.channel action=resp.action id=resp.id
                catch e
                    @warn "[CLIENT] Failed to decode as protobuf" exception=e
                    @info "[CLIENT] Raw data (hex)" data=bytes2hex(data[1:min(64, length(data))])
                end
            else
                @info "[CLIENT] No data yet (attempt $i/10)"
            end
            sleep(1)
        end

        @info "[CLIENT] Done listening"
    end
end
