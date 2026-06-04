#!/bin/bash

# Start Julia WS server
echo "Starting Julia WS server on port ${WsPort:-8002} on host ${WsHost:-0.0.0.0} ..."
WsHost=${WsHost:-0.0.0.0} WsPort=${WsPort:-8002} julia -t auto --project=/usr/app/research/fino /usr/app/research/fino/src/jobs/run_ws.jl &

# Start Julia ZMQ pricer
echo "Starting Julia ZMQ pricer on port ${PricerPort:-8102} on host ${PricerHost:-0.0.0.0} ..."
julia -t auto --project=/usr/app/research/fino /usr/app/research/fino/src/jobs/run_zmq_pricer.jl &


# Wait for all background processes to finish
wait -n

# Exit with the status of the first process that terminated
exit $?
