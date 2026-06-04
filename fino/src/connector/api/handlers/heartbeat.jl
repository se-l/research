using ProtoBuf
using HTTP
using HTTP.WebSockets: WebSocket
import HTTP.WebSockets

using ..WS: ChannelPb, ActionPb, MessagePb, pb2bytes

# Track heartbeat_started status for each websocket
# (No longer used with centralized heartbeat manager)

# Active connections and subscriptions (these would come from your manager)
const active_connections = Set{WebSocket}()
const subscriptions = Dict{Int, Set{WebSocket}}()
const HEARTBEAT_TASKS = Set{Task}()
const hb_lock = ReentrantLock()
const heartbeat_manager_task = Ref{Union{Task, Nothing}}(nothing)

function get_client_id(websocket::WebSocket)::String
    # Implement based on your requirements
    return string(websocket)
end

# Handle incoming heartbeat messages
function handle_on_heartbeat(websocket::WebSocket, msg::MessagePb)
    ensure_heartbeat_manager_running()
    
    lock(hb_lock) do
        if msg.action == ActionPb.SUBSCRIBE
            # Subscribe the websocket to the channel
            push!(active_connections, websocket)
            ch_key = Int(ChannelPb.HB)
            if !haskey(subscriptions, ch_key)
                subscriptions[ch_key] = Set{WebSocket}()
            end
            push!(subscriptions[ch_key], websocket)

        elseif msg.action == ActionPb.UNSUBSCRIBE
            # Unsubscribe the websocket from the channel
            ch_key = Int(ChannelPb.HB)
            if haskey(subscriptions, ch_key)
                delete!(subscriptions[ch_key], websocket)
            end
            delete!(active_connections, websocket)
        end
    end
end

function ensure_heartbeat_manager_running()
    if heartbeat_manager_task[] === nothing || istaskdone(heartbeat_manager_task[])
        heartbeat_manager_task[] = spawn_task(HEARTBEAT_TASKS, "HeartbeatManager") do
            heartbeat_manager_loop()
        end
    end
end

function heartbeat_manager_loop()
    @info "HeartbeatManager started"
    while true
        sleep(1)
        
        subs = []
        lock(hb_lock) do
            if haskey(subscriptions, Int(ChannelPb.HB))
                subs = collect(subscriptions[Int(ChannelPb.HB)])
            end
        end
        
        for ws in subs
            try
                if !ws.writeclosed && !ws.readclosed
                    # Send with timeout to prevent blocking
                    @async begin
                        send_heartbeat(ws)
                    end
                end
            catch e
                @warn "HeartbeatManager: Error sending heartbeat" exception=e
            end
        end
    end
end

# Send a heartbeat message to the client
function send_heartbeat(websocket::WebSocket)
    response = MessagePb(ChannelPb.HB, "hb", ActionPb.SUBSCRIBE, UInt8[])
#    @info "HeartbeatManager: send_heartbeat"
    lock(WS_SEND_LOCK) do
        HTTP.WebSockets.send(websocket, pb2bytes(response))
    end
end