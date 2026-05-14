using ProtoBuf

proto_files = [
    "Common.proto",
    "Command.proto",
    "PfRiskScenarios.proto",
    "RequestKalmanInit.proto",
    "RequestSSVICalibration.proto",
    "RequestTargetPortfolios.proto",
    "StressTestDs.proto",
    "Websocket.proto",
]
out_dir = raw"C:\repos\research\fino\src\connector\api\protos"
proto_dir = raw"C:\repos\trade\src\connector\api_minlp\protos"

for proto_file in proto_files
    println("Generating Julia: $proto_file")
    protojl(proto_file, proto_dir, out_dir)
end

println("All protos generated successfully!")