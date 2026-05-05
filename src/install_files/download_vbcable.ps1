# ============================================================================
# VB-Cable Download Script
# Downloads VB-Audio Cable installer package
# ============================================================================

param(
    [string]$DownloadPath = "$env:TEMP\VBCABLE_Driver_Pack45.zip"
)

$url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"

try {
    Write-Host "Downloading VB-Cable from $url..."
    Write-Host "Destination: $DownloadPath"

    # Download with progress
    $ProgressPreference = 'SilentlyContinue'  # Faster download
    Invoke-WebRequest -Uri $url -OutFile $DownloadPath -UseBasicParsing

    if (Test-Path $DownloadPath) {
        $fileSize = (Get-Item $DownloadPath).Length / 1MB
        Write-Host "Download complete! Size: $([math]::Round($fileSize, 2)) MB"
        exit 0
    } else {
        Write-Error "Download failed - file not found"
        exit 1
    }

} catch {
    Write-Error "Error downloading VB-Cable: $_"
    exit 1
}
