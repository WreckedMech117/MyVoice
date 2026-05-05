# ============================================================================
# VB-Audio Virtual Cable Detection Script
# Checks if VB-Audio Cable or VoiceMeeter Banana is installed
# Returns: Exit code 0 if found, 1 if not found
# ============================================================================

try {
    # Get all sound devices
    $soundDevices = Get-CimInstance Win32_SoundDevice -ErrorAction Stop

    # Check for VB-Audio devices
    # Look for: "VB-Audio Virtual Cable", "VB-Audio VoiceMeeter", "CABLE Output", etc.
    $vbAudioDevices = $soundDevices | Where-Object {
        $_.Name -match "VB-Audio" -or
        $_.Manufacturer -match "VB-Audio" -or
        $_.Name -match "CABLE (Input|Output)" -or
        $_.Name -match "VoiceMeeter"
    }

    if ($vbAudioDevices) {
        Write-Host "Virtual audio cable detected:"
        $vbAudioDevices | ForEach-Object {
            Write-Host "  - $($_.Name) ($($_.Manufacturer))"
        }
        exit 0  # Found
    } else {
        Write-Host "No virtual audio cable detected"
        exit 1  # Not found
    }

} catch {
    Write-Error "Error checking for virtual cable: $_"
    exit 1  # Error = treat as not found
}
