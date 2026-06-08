Set-Location "C:\"
# Build Docker image
docker buildx build `
    -f "./repos/research/docker/DockerfileApiBase" `
    -t "sebastianluen/mindbase:latest" `
    .

Write-Host "Build completed. Press any key to exit..."
#[Console]::ReadKey()