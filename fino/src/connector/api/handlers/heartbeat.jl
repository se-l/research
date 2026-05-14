using ProtoBuf
using HTTP
using HTTP.WebSockets: WebSocket, send

using ..WS: ChannelPb, ActionPb, MessagePb, pb2bytes

# Track heartbeat_started status for each websocket
const heartbeat_started = Dict{WebSocket, Bool}()

# Active connections and subscriptions (these would come from your manager)
const active_connections = Set{WebSocket}()
const subscriptions = Dict{Int, Set{WebSocket}}()

const HEARTBEAT_TASKS = Set{Task}()

function get_client_id(websocket::WebSocket)::String
    # Implement based on your requirements
    return string(websocket)
end

# Handle incoming heartbeat messages
function handle_on_heartbeat(websocket::WebSocket, msg::MessagePb)
    if msg.action == ActionPb.SUBSCRIBE
        # Subscribe the websocket to the channel
        push!(active_connections, websocket)
        ch_key = Int(ChannelPb.HB)
        if !haskey(subscriptions, ch_key)
            subscriptions[ch_key] = Set{WebSocket}()
        end
        push!(subscriptions[ch_key], websocket)

        # Start the heartbeat if it hasn't been started already
        if !get(heartbeat_started, websocket, false)
            spawn_task(HEARTBEAT_TASKS, "Heartbeat") do
                heartbeat(websocket)
            end
            heartbeat_started[websocket] = true
        end

    elseif msg.action == ActionPb.UNSUBSCRIBE
        # Unsubscribe the websocket from the channel
        ch_key = Int(ChannelPb.HB)
        if haskey(subscriptions, ch_key)
            delete!(subscriptions[ch_key], websocket)
        end
        delete!(active_connections, websocket)

        # Remove from heartbeat tracking
        if haskey(heartbeat_started, websocket)
            delete!(heartbeat_started, websocket)
        end
    end
end

# Send a heartbeat message to the client
function send_heartbeat(websocket::WebSocket)
    response = MessagePb(ChannelPb.HB, "hb", ActionPb.SUBSCRIBE, UInt8[])
    HTTP.WebSockets.send(websocket, pb2bytes(response))
end

# Continuously send heartbeats to the client
function heartbeat(websocket::WebSocket)
    @info "Start sending heartbeats to client: $(get_client_id(websocket))"

    while websocket in active_connections && haskey(subscriptions, Int(ChannelPb.HB)) && websocket in subscriptions[Int(ChannelPb.HB)]
        if websocket.writeclosed || websocket.readclosed
            @info "Heartbeat.heartbeat(): Client disconnected, stopping heartbeats"
            break
        end
        try
            send_heartbeat(websocket)
        catch e
            @error "Heartbeat.heartbeat(): $e"
            break
        end
        sleep(1)
    end

    # Clean up heartbeat tracking
    if haskey(heartbeat_started, websocket)
        delete!(heartbeat_started, websocket)
    end

    @info "Heartbeat.heartbeat(): Stopped sending heartbeats to client: $(get_client_id(websocket))"
end