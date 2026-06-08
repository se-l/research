module MQMsgBroker

using AMQPClient
using ProtoBuf

using Fino.WS   # MessagePb, ChannelPb, parse_pb, handler modules

export start_broker

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEFAULT_HOST     = get(ENV, "MQHost", "localhost")
const DEFAULT_PORT     = 5672
const DEFAULT_VHOST    = get(ENV, "MQVirtualHost", "dev")
const DEFAULT_USER     = get(ENV, "MQUser", "myuser")
const DEFAULT_PASS     = get(ENV, "MQPassword", "mysecurepassword")

const EXCHANGE_IN      = "calc_engine"            # topic — inbound requests
const REQUEST_QUEUE    = "calc_engine.inbound"    # single work queue
const BINDING_KEY      = "calc.#"                 # matches calc.target_portfolio, calc.kalman_init, …
const PREFETCH_COUNT   = 32                       # buffered unacked msgs; enables conflation

# ═══════════════════════════════════════════════════════════════════════════════
# Conflating queue — same invariant as WsMsgBroker
#
# Per channel slot:
#   • pending  – the latest (msg, ctx) pair, or nothing
#   • running  – true while a worker task is alive
#
# At most one worker per channel.  New messages overwrite pending.
# ═══════════════════════════════════════════════════════════════════════════════

struct PendingItem
    msg :: Any
    ctx :: MQContext
end

mutable struct ConflatingQueue
    pending ::Union{Nothing, PendingItem}
    running ::Bool
    lock    ::ReentrantLock
    ConflatingQueue() = new(nothing, false, ReentrantLock())
end

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
    enqueue_and_process(msg, ctx, handler_fn)

Drop `msg` into the conflating queue for `msg.channel`.
If no worker is running for that channel, spawn one.

`handler_fn` must accept `(ctx::MQContext, msg)`.
"""
function enqueue_and_process(msg, ctx::MQContext, handler_fn)
    q = _get_or_create_queue(msg.channel)

    should_spawn = lock(q.lock) do
        q.pending = PendingItem(msg, ctx)   # overwrite — conflation happens here
        if q.running
            false
        else
            q.running = true
            true
        end
    end

    should_spawn || return

    Threads.@spawn begin
        try
            while true
                local item = lock(q.lock) do
                    m = q.pending
                    q.pending = nothing
                    m
                end

                item === nothing && break

                try
                    handler_fn(item.ctx, item.msg)
                catch e
                    @error "[MQMsgBroker] handler error" channel=item.msg.channel exception=(e, catch_backtrace())
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

# ═══════════════════════════════════════════════════════════════════════════════
# Channel → handler dispatch
# ═══════════════════════════════════════════════════════════════════════════════

function _route_channel(channel)
    if channel == ChannelPb.TARGET_PORTFOLIO
        TargetPortfoliosHandler.handle_on_msg
    elseif channel == ChannelPb.STRESS_TEST_DS
        StressTestDsHandler.handle_on_msg
    elseif channel == ChannelPb.KALMAN_INIT
        KalmanHandler.handle_on_msg
    elseif channel == ChannelPb.REQUEST_SSVI_CALIBRATION
        SSVICalibrationHandler.handle_on_msg
    elseif channel == ChannelPb.REQUEST_PF_RISK_SCENARIOS
        PfRiskScenarioHandler.handle_on_msg
    else
        @warn "[MQMsgBroker] Unknown channel" channel
        nothing
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Broker entry point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    start_broker(; host, port, vhost, user, password) -> Nothing

Connect to RabbitMQ, declare the topology, and enter the consume loop.
Blocks until the connection is closed or interrupted.
"""
function start_broker(;
        host     = DEFAULT_HOST,
        port     = DEFAULT_PORT,
        vhost    = DEFAULT_VHOST,
        user     = DEFAULT_USER,
        password = DEFAULT_PASS,
    )::Nothing

    @info "[MQMsgBroker] Connecting to RabbitMQ" host port vhost

    auth_params = Dict{String,Any}(
            "MECHANISM" => "AMQPLAIN",
            "LOGIN"     => user,
            "PASSWORD"  => password,
        )

    conn = AMQPClient.connection(; virtualhost=vhost, host=host, port=port, auth_params=auth_params)
    chan = AMQPClient.channel(conn, 1, true)   # channelid=1, connect=true

    # ── Declare topology ──────────────────────────────────────────────────────
    AMQPClient.exchange_declare(chan, EXCHANGE_IN,  "topic"; durable = true)
    AMQPClient.exchange_declare(chan, EXCHANGE_OUT, "topic"; durable = true)
    AMQPClient.queue_declare(chan, REQUEST_QUEUE; durable = true)
    AMQPClient.queue_bind(chan, REQUEST_QUEUE, EXCHANGE_IN, BINDING_KEY)
    AMQPClient.basic_qos(chan, 0, PREFETCH_COUNT, false)

    mq = MQConnection(chan, ReentrantLock())

    @info "[MQMsgBroker] Consuming from" queue=REQUEST_QUEUE exchange_in=EXCHANGE_IN exchange_out=EXCHANGE_OUT

    # ── Consume loop ──────────────────────────────────────────────────────
    consumer_fn = (msg) -> begin
        props      = msg.properties
        body       = msg.data

        prop_dict = Dict{String,Any}(
            "correlation_id" => hasproperty(props, :correlation_id) ? string(props.correlation_id) : "",
            "reply_to"       => hasproperty(props, :reply_to)       ? string(props.reply_to)       : "",
        )
        try
            if isempty(body)
                @warn "[MQMsgBroker] Received empty message"
                AMQPClient.basic_ack(chan, msg.delivery_tag)
                return
            end

            msg_pb = parse_pb(body, MessagePb)

            client_reply_to = get(prop_dict, "reply_to", "")
            reply_to = isempty(client_reply_to) ?
                "response.$(lowercase(string(msg_pb.channel)))" :
                client_reply_to

            ctx = MQContext(mq, reply_to, prop_dict)

            @debug "[MQMsgBroker] Received" channel=msg_pb.channel action=msg_pb.action id=msg_pb.id

            if msg_pb.channel == ChannelPb.HB
                handle_on_heartbeat(ctx, msg_pb)
                AMQPClient.basic_ack(chan, msg.delivery_tag)
                return
            end

            handler_fn = _route_channel(msg_pb.channel)
            AMQPClient.basic_ack(chan, msg.delivery_tag)

            if handler_fn !== nothing
                enqueue_and_process(msg_pb, ctx, handler_fn)
            end

        catch e
            @error "[MQMsgBroker] Consume callback error" exception=(e, catch_backtrace())
            try
                AMQPClient.basic_nack(chan, deliver.delivery_tag; requeue = false)
            catch; end
        end
    end

    AMQPClient.basic_consume(chan, REQUEST_QUEUE, consumer_fn; no_ack = false)

    @info "[MQMsgBroker] Broker running"
    try
        while isopen(conn)
            sleep(1)
        end
    catch e
        e isa InterruptException || rethrow()
        @info "[MQMsgBroker] Interrupted, closing"
        AMQPClient.close(chan)
        AMQPClient.close(conn)
    end
    @info "[MQMsgBroker] Connection closed" state=conn.state

    # Best-effort: let any running workers finish (up to 30 s)
    deadline = time() + 30.0
    while time() < deadline
        any_running = lock(_registry_lock) do
            any(q -> lock(q.lock) do; q.running; end, values(_queues))
        end
        any_running || break
        sleep(0.1)
    end

    _remove_all_queues!()
    @info "[MQMsgBroker] Broker stopped"
    return nothing
end

end # module MQMsgBroker