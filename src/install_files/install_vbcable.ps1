# ============================================================================
# VB-Cable Installation Script
# Extracts and installs VB-Audio Cable driver
# ============================================================================

param(
    [string]$ZipPath = "$env:TEMP\VBCABLE_Driver_Pack45.zip",
    [string]$ExtractPath = "$env:TEMP\VBCABLE"
)

try {
    # Extract ZIP
    Write-Host "Extracting VB-Cable installer..."
    if (Test-Path $ExtractPath) {
        Remove-Item -Recurse -Force $ExtractPath
    }
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force

    # Determine architecture and installer
    if ([Environment]::Is64BitOperatingSystem) {
        $installerExe = Join-Path $ExtractPath "VBCABLE_Setup_x64.exe"
        Write-Host "64-bit system detected, using VBCABLE_Setup_x64.exe"
    } else {
        $installerExe = Join-Path $ExtractPath "VBCABLE_Setup.exe"
        Write-Host "32-bit system detected, using VBCABLE_Setup.exe"
    }

    if (-not (Test-Path $installerExe)) {
        Write-Error "Installer not found: $installerExe"
        exit 1
    }

    # Run installer
    # Note: VB-Cable installer requires admin rights and may show UAC prompt
    Write-Host "Running VB-Cable installer..."
    Write-Host "IMPORTANT: The installer will require administrator approval (UAC prompt)"

    $process = Start-Process -FilePath $installerExe -Wait -PassThru -Verb RunAs

    if ($process.ExitCode -eq 0) {
        Write-Host "VB-Cable installation completed successfully"
        Write-Host "REBOOT REQUIRED: Please restart your PC for VB-Cable to work"
        exit 0
    } else {
        Write-Warning "VB-Cable installer exited with code: $($process.ExitCode)"
        exit $process.ExitCode
    }

} catch {
    Write-Error "Error installing VB-Cable: $_"
    exit 1
} finally {
    # Cleanup
    if (Test-Path $ExtractPath) {
        Remove-Item -Recurse -Force $ExtractPath -ErrorAction SilentlyContinue
    }
}
