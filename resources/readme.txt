================================================================================
MyVoice V2 - Expressive Voice Communication for Everyone
Version 2.1.0
================================================================================

Thank you for choosing MyVoice!

MyVoice V2 is a powerful desktop application for high-quality text-to-speech
synthesis with embedded Qwen3-TTS, emotion control, Voice Design, voice cloning,
and dual audio routing capabilities.

================================================================================
SYSTEM REQUIREMENTS
================================================================================

Minimum Requirements:
- Windows 10 (64-bit) version 1809 or later
- Windows 11 (all editions supported)
- 8 GB RAM (16 GB recommended for smooth operation)
- 500 MB free disk space for MyVoice application
- 8 GB+ additional space for Qwen3-TTS models (downloaded on first use)
- Audio input/output devices (microphone and speakers/headphones)

Optional:
- Virtual audio cable (VB-Cable) for voice chat routing
- Internet connection for first-time model downloads

================================================================================
WHAT'S INCLUDED
================================================================================

MyVoice includes:
- Complete Python 3.10 runtime (no separate Python installation needed)
- PyQt6 GUI framework
- PyTorch (CPU-optimized for fast inference)
- OpenAI Whisper for speech recognition
- All required dependencies

Downloaded on First Use:
- Qwen3-TTS models (~3.4GB per model, downloaded on first TTS use)
- Whisper AI models (~140MB-3GB, downloaded on first transcription)

================================================================================
INSTALLATION INSTRUCTIONS
================================================================================

1. Run this installer (MyVoice-Setup-v2.1.0.exe)
2. Follow the installation wizard
3. Choose installation directory (default: C:\Program Files\MyVoice)
4. Select optional shortcuts (Desktop, Start Menu)
5. Click Install and wait for completion
6. Launch MyVoice from Start Menu or Desktop shortcut

First Run:
- On first launch, MyVoice will download the Whisper "base" model (~140MB)
- This is a one-time download and will be cached for future use
- Internet connection required for first-time model download
- Subsequent launches will use the cached model (no download needed)

================================================================================
VOICE CLONING (QWEN3-TTS)
================================================================================

MyVoice uses embedded Qwen3-TTS for high-quality voice cloning:

- Works completely offline after initial model download
- Clone any voice with just 3-15 seconds of audio
- Supports emotion control via natural language prompts
- Three model types: Base (clone), CustomVoice, VoiceDesign

First-time TTS use will download models (~3.4GB). This is automatic
and only happens once per model type.

================================================================================
QUICK START GUIDE
================================================================================

1. Launch MyVoice from your Start Menu or Desktop

2. Configure Audio Devices:
   - Go to Settings > Audio
   - Select your microphone input device
   - Select your output device (speakers/headphones)
   - Configure virtual microphone if needed

3. Set Up Voice Profile:
   - Create a voice profile with a 3-15 second audio sample
   - WAV format recommended, clear speech, minimal noise
   - Adjust emotion settings as desired

4. Start Using MyVoice:
   - Type text in the main window
   - Click "Generate Speech" to synthesize
   - Use "Quick Speak" for frequently used phrases
   - Enable background transcription for real-time speech-to-text

================================================================================
FEATURES
================================================================================

V2 Core Features:
- High-quality text-to-speech with embedded Qwen3-TTS
- 9 bundled voice timbres (Ryan, Vivian, Aiden, and more)
- 5 emotion presets (Neutral, Happy, Sad, Angry, Flirtatious)
- Voice Design: Create voices from text descriptions
- Voice Clone: Clone any voice from 3-10 second audio samples
- Quick Speak with keyboard shortcuts and voice/emotion overrides
- Dual audio routing for virtual microphone output
- Real-time speech recognition with Whisper

V2 Advanced Features:
- Custom emotion prompts for nuanced expression
- Emotion keyboard shortcuts (Ctrl+1 through Ctrl+5)
- Streaming TTS with ~97ms first-packet latency
- Lazy model loading for memory efficiency
- Virtual microphone output for streaming/gaming
- Microphone passthrough to virtual mic for mixed voice chat
- Audio device hot-swapping detection
- Settings persistence with auto-recovery

================================================================================
TROUBLESHOOTING
================================================================================

MyVoice won't start:
- Check Windows Event Viewer for error details
- Ensure you have Visual C++ Redistributable installed
- Try running as Administrator
- Check logs in: %LOCALAPPDATA%\MyVoice\logs

Whisper model download fails:
- Check internet connection
- Verify firewall isn't blocking Python
- Models download to: %USERPROFILE%\.cache\whisper\
- Try manually downloading from OpenAI Whisper repository

TTS model issues:
- First TTS use downloads models (~3.4GB) - check internet
- Models are cached in user directory after download
- Check logs for specific error messages

Audio device issues:
- Check Windows audio settings
- Verify devices are not in use by other applications
- Try restarting MyVoice
- Check device permissions in Windows Privacy settings

Performance issues:
- Close unnecessary applications
- Use smaller Whisper model (tiny or base)
- Disable background transcription if not needed
- Check system resources (Task Manager)

================================================================================
UNINSTALLATION
================================================================================

To uninstall MyVoice:

Method 1: Windows Settings
1. Open Windows Settings
2. Go to Apps > Installed apps
3. Find "MyVoice"
4. Click "Uninstall"

Method 2: Control Panel
1. Open Control Panel
2. Go to Programs > Uninstall a program
3. Select "MyVoice"
4. Click Uninstall

Method 3: Uninstaller
1. Go to Start Menu > MyVoice
2. Click "Uninstall MyVoice"

Note: Uninstallation will remove:
- MyVoice application files
- Start Menu and Desktop shortcuts
- Registry entries

Uninstallation will NOT remove:
- User settings and configurations
- Downloaded Whisper models (in user cache)
- User-created voice profiles

To fully remove all data:
- Delete: %LOCALAPPDATA%\MyVoice
- Delete: %APPDATA%\MyVoice
- Delete: %USERPROFILE%\.cache\whisper (if no other apps use it)

================================================================================
SUPPORT & FEEDBACK
================================================================================

For help and support:
- GitHub Issues: https://github.com/myvoice/myvoice/issues
- Documentation: https://github.com/myvoice/myvoice/wiki
- Email: support@myvoice.local

Report bugs:
- Include MyVoice version (2.1.0)
- Describe steps to reproduce
- Attach relevant log files from %LOCALAPPDATA%\MyVoice\logs

Feature requests:
- Submit via GitHub Issues
- Tag as "enhancement"
- Describe use case and benefits

================================================================================
LICENSE & COPYRIGHT
================================================================================

MyVoice is released under the MIT License
Copyright (c) 2025-2026 MyVoice Development Team

See LICENSE.txt for full license text and third-party notices.

================================================================================
GETTING STARTED
================================================================================

Ready to begin? Here's what to do next:

1. ✅ Complete this installation
2. ✅ Launch MyVoice from Start Menu
3. ✅ Wait for Whisper model download (first time only)
4. ✅ Configure audio devices in Settings
5. ✅ Create a voice profile with audio sample
6. ✅ Start synthesizing speech!

Enjoy using MyVoice!

================================================================================
