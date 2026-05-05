# MyVoice Troubleshooting Guide

This guide covers common issues and solutions for MyVoice V2.

## Quick Diagnostics

Before troubleshooting, gather this information:
1. Check `logs/myvoice.log` for error details
2. Note your Windows version (10/11) and build number
3. Check available RAM (16GB recommended)
4. Verify audio device connections

---

## Audio Issues

### No Sound Output

**Symptoms:** TTS generates but no audio plays

**Solutions:**
1. **Check monitor device selection:**
   - Open Settings (Ctrl+S) → Audio tab
   - Verify correct speaker device is selected
   - Click "Test" to verify audio output

2. **Check Windows audio:**
   - Right-click speaker icon in system tray
   - Open Sound Settings
   - Verify default playback device

3. **Restart audio services:**
   ```powershell
   # Run as Administrator
   net stop audiosrv && net start audiosrv
   ```

### Audio Device Not Found

**Symptoms:** "Audio device not available" error

**Solutions:**
1. Reconnect the audio device
2. Open Settings → Audio → Click "Refresh Devices"
3. Select a different device from the dropdown
4. Check Windows Device Manager for driver issues

### Error Code -9996 (PyAudio)

**Symptoms:** Audio playback fails with internal error

**Causes:**
- Audio device disconnected during playback
- Device sample rate mismatch
- Driver conflicts

**Solutions:**
1. Close and reopen MyVoice
2. Select "Windows WASAPI" host API in Settings
3. Update audio drivers
4. Try a different audio device

### Audio Crackling or Distortion

**Symptoms:** Playback has pops, clicks, or distortion

**Solutions:**
1. Increase audio buffer size in advanced settings
2. Close other audio applications
3. Disable audio enhancements in Windows Sound settings
4. Update audio drivers

---

## Virtual Microphone Issues

### Virtual Mic Not Detected

**Symptoms:** No virtual microphone devices shown in Settings

**Solutions:**
1. **Install VB-Audio Cable:**
   - Download from https://vb-audio.com/Cable/
   - Run installer as Administrator
   - **Restart your computer** (required!)

2. **Alternative: Voicemeeter:**
   - Download from https://vb-audio.com/Voicemeeter/
   - Run installer as Administrator
   - Restart computer

3. **Verify installation:**
   - Open Windows Sound Settings
   - Look for "CABLE Input" in Recording devices
   - Look for "CABLE Output" in Playback devices

### Virtual Mic Not Working in Apps

**Symptoms:** MyVoice outputs to virtual mic but Discord/Zoom doesn't receive

**Solutions:**
1. **Configure target application:**
   - Set input device to "CABLE Output" (not CABLE Input)
   - In Discord: User Settings → Voice & Video → Input Device

2. **Check routing:**
   - MyVoice sends to "CABLE Input" (output device)
   - Target app receives from "CABLE Output" (input device)

3. **Test with Windows Voice Recorder:**
   - Set recording device to "CABLE Output"
   - Generate speech in MyVoice
   - Verify audio is captured

### VB-Cable Driver Issues

**Symptoms:** VB-Cable device shows warning in Device Manager

**Solutions:**
1. Reinstall VB-Cable:
   ```powershell
   # Run as Administrator in VB-Cable folder
   .\VBCABLE_Setup_x64.exe -uninstall
   # Restart
   .\VBCABLE_Setup_x64.exe -install
   # Restart again
   ```

2. Check for driver updates at https://vb-audio.com/Cable/

---

## TTS / Model Issues

### Model Loading Failure

**Symptoms:** "Failed to load voice model" error

**Solutions:**
1. **Check disk space:**
   - Each model requires ~3.4GB
   - Verify 10GB+ free space

2. **Check internet connection:**
   - Models download from HuggingFace on first use
   - Verify https://huggingface.co is accessible

3. **Clear model cache:**
   ```powershell
   # Delete cached models (will re-download)
   Remove-Item -Recurse ~\.cache\huggingface\hub\models--Qwen*
   ```

4. **Check firewall/proxy:**
   - Allow Python to access the internet
   - Configure proxy in environment variables if needed

### Out of Memory Error

**Symptoms:** "Not enough memory to load voice model" or application crash

**Solutions:**
1. **Close other applications:**
   - Each model uses ~3.4GB RAM
   - Close browsers, games, other memory-heavy apps

2. **Check available RAM:**
   - Open Task Manager (Ctrl+Shift+Esc)
   - Memory should show 4GB+ available

3. **Increase virtual memory:**
   - System Properties → Advanced → Performance Settings
   - Advanced → Virtual memory → Change
   - Set to at least 8GB

4. **Use fewer models:**
   - Stick to one voice type to avoid model switching
   - Bundled voices use CustomVoice model only

### Speech Generation Timeout

**Symptoms:** "Speech generation timed out" error

**Solutions:**
1. **Shorten text:**
   - Split long text into shorter segments
   - Recommended: Under 200 words per generation

2. **Check CPU usage:**
   - High CPU usage slows generation
   - Close background applications

3. **Wait for initial load:**
   - First generation is slower (model initialization)
   - Subsequent generations are faster

### Voice Sounds Wrong

**Symptoms:** Voice doesn't match expectations

**Solutions:**
1. **For cloned voices:**
   - Use 5-10 second clear speech samples
   - Avoid background noise in source audio
   - Provide accurate transcription

2. **For designed voices:**
   - Keep descriptions under 100 words
   - Be specific about age, gender, accent
   - Avoid contradictory descriptions

3. **Regenerate:**
   - VoiceDesign has natural variation
   - Try generating again for different results

---

## UI Issues

### Window Not Appearing

**Symptoms:** Application starts but window is not visible

**Solutions:**
1. **Check system tray:**
   - MyVoice may be minimized to tray
   - Click tray icon to restore

2. **Reset window position:**
   - Delete `settings.json` to reset window geometry
   - Or edit `settings.json`, set `"window_geometry": null`

3. **Check for off-screen window:**
   - Disconnect external monitors
   - Press Win+D to show desktop
   - Press Alt+Space, then M, then arrow keys

### System Tray Icon Missing

**Symptoms:** Window minimizes but no tray icon appears

**Solutions:**
1. **Check Windows tray settings:**
   - Settings → Personalization → Taskbar
   - System tray → Show all icons

2. **Restart Explorer:**
   ```powershell
   taskkill /f /im explorer.exe && start explorer.exe
   ```

### Blurry Text / Scaling Issues

**Symptoms:** Text appears blurry or controls are wrong size

**Solutions:**
1. **Configure per-app DPI:**
   - Right-click MyVoice.exe → Properties
   - Compatibility → Change high DPI settings
   - Override scaling: Application

2. **Check Windows scaling:**
   - Settings → Display → Scale
   - Try 100%, 125%, or 150%

---

## Whisper / Transcription Issues

### Transcription Fails

**Symptoms:** "Transcription failed" error for voice samples

**Solutions:**
1. **Check audio format:**
   - Must be WAV format
   - 16-bit, mono preferred
   - Sample rate: 16kHz or higher

2. **Check audio quality:**
   - Clear speech without background noise
   - Duration: 3-30 seconds

3. **Memory for Whisper:**
   - Whisper needs ~1GB additional RAM
   - Close other apps if memory is low

### Inaccurate Transcription

**Symptoms:** Transcription text doesn't match audio

**Solutions:**
1. **Use higher quality audio:**
   - Record in quiet environment
   - Use consistent microphone distance

2. **Manually correct:**
   - Edit the `.txt` file alongside the `.wav`
   - Transcription is optional for cloning

---

## Performance Issues

### High CPU Usage

**Symptoms:** CPU stays at 100% or system is slow

**Solutions:**
1. **Normal during generation:**
   - TTS is CPU-intensive
   - Usage should drop after generation

2. **If persistent:**
   - Close and restart MyVoice
   - Check for stuck background threads in Task Manager

### Slow Startup

**Symptoms:** Application takes long to start

**Solutions:**
1. **First launch is slow:**
   - Models download on first use
   - Subsequent launches are faster

2. **Reduce startup load:**
   - Delete unused voice files
   - Clear old design sessions

3. **SSD recommended:**
   - Model loading is I/O intensive
   - Moving cache to SSD helps

### High Memory Usage

**Symptoms:** MyVoice uses 4GB+ RAM

**Solutions:**
1. **Expected behavior:**
   - Qwen3-TTS models are ~3.4GB each
   - Only one model loaded at a time

2. **Reduce memory:**
   - Avoid switching voice types frequently
   - Close and restart periodically

---

## Installation Issues

### PyAudio Installation Failed

**Symptoms:** Error during pip install pyaudio

**Solutions:**
1. **Windows Build Tools:**
   - Install Visual C++ Build Tools
   - Or use pre-built wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/

2. **Use pre-built package:**
   ```powershell
   pip install pipwin
   pipwin install pyaudio
   ```

### Missing DLL Errors

**Symptoms:** "DLL not found" on startup

**Solutions:**
1. **Install Visual C++ Redistributable:**
   - Download from Microsoft
   - Install both x64 and x86 versions

2. **Reinstall application:**
   - Uninstall completely
   - Delete application folder
   - Reinstall fresh

---

## Error Code Reference

| Code | Category | Description | Solution |
|------|----------|-------------|----------|
| `TTS_MODEL_LOAD_FAILED` | TTS | Model failed to load | Check memory, disk, internet |
| `TTS_GENERATION_TIMEOUT` | TTS | Generation exceeded time limit | Shorten text |
| `TTS_EMPTY_TEXT` | TTS | No text provided | Enter text before generating |
| `AUDIO_DEVICE_NOT_FOUND` | Audio | Selected device unavailable | Reconnect or select different device |
| `AUDIO_STREAM_ERROR` | Audio | Playback stream failed | Restart application |
| `VOICE_FILE_INVALID` | Voice | Voice file corrupted or invalid | Re-record or delete voice |
| `VOICE_TOO_SHORT` | Voice | Audio under 3 seconds | Use longer sample |
| `CONFIG_CORRUPTED` | Config | Settings file corrupted | Automatic recovery to defaults |

---

## Getting Help

If issues persist:

1. **Check logs:**
   - Location: `logs/myvoice.log`
   - Set log level to DEBUG for more detail

2. **Gather information:**
   - Windows version and build
   - MyVoice version (shown in title bar)
   - Error messages from log file

3. **Report issues:**
   - GitHub: https://github.com/myvoice/myvoice/issues
   - Include: Steps to reproduce, log excerpts, system info

---

*Last Updated: 2026-02-08*
*MyVoice V2.0.0*
