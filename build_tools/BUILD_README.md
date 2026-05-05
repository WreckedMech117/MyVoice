# MyVoice V2 Build Instructions

## Quick Build

Simply run the build script:
```bash
build_tools\build.bat
```

This will:
1. Run PyInstaller to bundle the application
2. Copy voice_files to the distribution root
3. Create necessary directories
4. Display build summary

## Distribution Structure

After building, your distribution will look like this:

```
dist/MyVoice/
├── MyVoice.exe              # Main executable (~24 MB)
├── _internal/               # Python runtime and dependencies (~300 MB)
│   ├── ffmpeg/              # Bundled ffmpeg binaries for Whisper transcription
│   │   ├── ffmpeg.exe       # Audio/video processing tool
│   │   └── ffprobe.exe      # Media stream analyzer
│   └── ...                  # Other Python dependencies
├── voice_files/             # Voice samples and user data
│   ├── embeddings/          # Saved voice embeddings (cloned/designed voices)
│   └── design_sessions/     # Voice Design working files
├── config/                  # Created on first run (settings.json)
└── logs/                    # Application logs
```

**Important V2 Notes:**
- Qwen3-TTS models (~3.4GB each) download on first use, NOT bundled
- Only ONE model loads at a time (lazy loading for memory efficiency)
- Whisper models (~1-3GB) also download on first use

Total distribution size: ~300 MB (before model downloads)

### FFmpeg Integration

The application bundles ffmpeg binaries required for Whisper transcription functionality:
- **ffmpeg.exe**: Required by OpenAI Whisper for audio file processing
- **ffprobe.exe**: Used for media file analysis
- These are located in the `ffmpeg/` directory in the project root
- The bundled ffmpeg is added to PATH at runtime for Windows 10 and Windows 11 compatibility
- **Note**: If you need to update ffmpeg, copy the new binaries to the `ffmpeg/` directory before building

## Testing the Build

To test the built executable:

```bash
cd dist\MyVoice
MyVoice.exe
```

**Important:** Always run from the `dist\MyVoice` directory so the application can find voice_files/ and config/ directories.
