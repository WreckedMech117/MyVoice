# MyVoice - Voice Cloning Desktop Application

A desktop application for voice cloning and text-to-speech using GPT-SoVITS technology. MyVoice provides an intuitive PyQt6 interface for creating and using custom voice profiles with fast, localized audio processing.

## Features

- **Voice Cloning**: Create custom voice profiles from audio samples
- **Real-time TTS**: Generate speech with custom voice profiles
- **Audio Transcription**: Automatic transcription using OpenAI Whisper
- **Virtual Microphone Support**: Route generated audio to communication apps
- **Audio Device Management**: Comprehensive audio device control
- **Modern UI**: Clean, responsive PyQt6 interface

## System Requirements

- **Operating System**: Windows 10/11 (64-bit)
- **Memory**: 8GB RAM minimum (16GB recommended for large models)
- **Storage**: 12GB free space (for dependencies, GPT-SoVITS with models, and cache)
- **Python**: 3.10 or higher

### Included Dependencies

- **FFmpeg** - Audio processing for Whisper (included in `ffmpeg/` directory)
  - Pre-configured for Windows
  - Used by OpenAI Whisper for audio transcription

### Required External Software

- **GPT-SoVITS** - Voice cloning engine (not included in this repository)
  - Download and setup instructions: [GPT-SoVITS GitHub](https://github.com/RVC-Boss/GPT-SoVITS)

- **VB-Audio Cable** (for Virtual Microphone - Simple)
  - Download: [VBCABLE_Driver_Pack45.zip](https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip)

  OR

- **VoiceMeeter Banana** (for Virtual Microphone - Advanced with full audio control)
  - Download: [VoicemeeterSetup](https://download.vb-audio.com/Download_CABLE/VoicemeeterSetup_v2119.zip)

- **Microsoft Visual C++ Redistributable** (usually pre-installed)
  - Download: [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/WreckedMech117/MyVoice.git
cd MyVoice
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Setup GPT-SoVITS

Download and setup [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) separately. By default, MyVoice expects it running at `http://127.0.0.1:9880`.

## Usage

### Running MyVoice

```bash
python -m myvoice.main
```

Or if installed via pip:

```bash
myvoice
```

### First Time Setup

1. **Start GPT-SoVITS Server**
   - Ensure GPT-SoVITS is running (default: http://127.0.0.1:9880)
   - Configure server URL in MyVoice Settings if needed

2. **Configure Audio Devices**
   - Select your monitor speaker for audio output
   - Select virtual microphone device:
     - "CABLE Input" for VB-Audio Cable users
     - "VoiceMeeter Input" or "VoiceMeeter Aux Input" for VoiceMeeter users

3. **Create Voice Profile**
   - Record or import audio sample (10 seconds maximum, WAV format)
   - Use "Transcribe" to generate transcription
   - Select and use your custom voice profile!

### Creating Voice Profiles

**Option 1: Using WAV + Text File**
1. Place a `.wav` file (max 10 seconds) in the `voice_files/` folder
2. Create a `.txt` file with the same name containing the transcription
3. Profile will appear in the dropdown

**Option 2: Using Transcribe Button**
1. Place a `.wav` file in `voice_files/` folder
2. Use the "Transcribe" button in MyVoice
3. Transcription will be generated automatically using Whisper

### Text-to-Speech

1. Select voice profile from dropdown
2. Enter text in the input field
3. Click **"Speak"** or use Quick Speak dialog
4. Audio plays through selected monitor device

### Virtual Microphone

1. Enable virtual microphone in settings
2. Test with "Test Virtual" button
   - Note: You won't hear this test unless using VoiceMeeter
3. Configure your communication app (Discord, Zoom, etc.) to use the virtual mic
4. Generated speech routes to apps in real-time

## Project Structure

```
MyVoice/
├── src/
│   └── myvoice/           # Application source code
│       ├── main.py        # Entry point
│       ├── app.py         # Application controller
│       ├── ui/            # User interface components
│       ├── services/      # Business logic services
│       ├── models/        # Data models
│       └── utils/         # Utility functions
├── ffmpeg/                # FFmpeg binaries for audio processing
├── voice_files/           # Voice profile samples
├── logs/                  # Application logs
├── config/                # Configuration files
├── requirements.txt       # Python dependencies
├── setup.py               # Package configuration
├── LICENSE                # License file
└── README.md              # This file
```

## Dependencies

### Core Dependencies
- **PyQt6** (≥6.6.0) - GUI framework
- **requests** (≥2.31.0) - HTTP client
- **openai-whisper** (≥20231117) - Speech recognition
- **torch** (≥2.0.0) - Machine learning framework
- **pyaudio** (≥0.2.13) - Audio I/O
- **numpy** (≥1.24.0) - Numerical processing

### Development Dependencies
- **pytest** (≥7.4.0) - Testing framework
- **pytest-qt** (≥4.2.0) - PyQt testing
- **pytest-asyncio** (≥0.21.0) - Async testing

## Configuration

Application settings are stored in:
- **Windows**: `%APPDATA%\MyVoice\settings.json`
- **Linux**: `~/.config/MyVoice/settings.json`
- **Mac**: `~/Library/Application Support/MyVoice/settings.json`

## Logs

Application logs are written to:
- **Location**: `logs/myvoice.log`
- **Rotation**: Automatic when file exceeds size limit

## Troubleshooting

### Installation Issues

**PyAudio Installation Fails**
- Install Microsoft Visual C++ Build Tools
- Download: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**PyTorch Installation Slow**
- PyTorch is ~2GB download
- Ensure stable internet connection
- Installation may take 5-10 minutes

### Runtime Issues

**Application Won't Start**
- Check `logs/myvoice.log` for errors
- Ensure all dependencies are installed
- Verify Python 3.10+ is being used

**No Audio Output**
- Verify audio device selection in Settings
- Check Windows audio settings
- Ensure device isn't in use by another application

**GPT-SoVITS Connection Failed**
- Start GPT-SoVITS server before running MyVoice
- Verify server URL in Settings (default: http://127.0.0.1:9880)
- Check that firewall isn't blocking the connection

**Transcription Fails**
- Whisper downloads models on first use (~1-3GB)
- Ensure internet connection is available
- Models are cached for offline use after download

## Development

### Running Tests

```bash
pytest
```

### Installing in Development Mode

```bash
pip install -e .
```

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

## FAQ

**Q: Does this work with any communication app?**
A: Yes, with apps that support virtual audio devices (Discord, Zoom, Teams, Skype, etc.)

**Q: Can I use my own audio samples?**
A: Yes! WAV format recommended. Samples should be 5-10 seconds of clear speech.

**Q: How much disk space do models need?**
A: Whisper models: 1-3GB, PyTorch: ~2GB, GPT-SoVITS models: ~6GB. Total ~10-12GB.

**Q: Does this work offline?**
A: After initial model downloads, transcription works offline. GPT-SoVITS must run locally.

**Q: Where is GPT-SoVITS included?**
A: GPT-SoVITS is a separate dependency and not included in this repository. Download it separately from the [GPT-SoVITS GitHub](https://github.com/RVC-Boss/GPT-SoVITS).

## License

See [LICENSE](LICENSE) file for details.

## Credits

- **GPT-SoVITS**: Voice cloning technology
- **OpenAI Whisper**: Speech recognition
- **PyQt6**: GUI framework

## Version

Current version: **1.0.0**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on the [GitHub repository](https://github.com/WreckedMech117/MyVoice/issues).
