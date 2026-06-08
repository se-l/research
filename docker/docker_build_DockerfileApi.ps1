Set-Location "C:\"

docker buildx build `
    -f "./repos/research/docker/DockerfileApi" `
    -t "sebastianluen/mind:dev" `
    .

Write-Host "Build completed. Press any key to exit..."
#[Console]::ReadKey()