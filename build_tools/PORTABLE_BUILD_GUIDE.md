# MyVoice V2 Portable Build Guide

This guide explains how to build, test, and distribute MyVoice V2 as a fully portable application.

## Overview

The portable build creates a self-contained distribution where:
- **No installation required** - Users extract and run
- **All data stored locally** - Settings, logs, and voice files stay in the app folder
- **No registry changes** - No system modifications
- **Fully movable** - Can be copied to USB drives or different locations
- **No AppData usage** - Everything is in the application directory

## V2 Features

- **Embedded Qwen3-TTS** - Voice cloning and generation (models download on first use)
- **Emotion Control** - 5 presets + custom emotion prompts
- **Voice Design** - Create voices from text descriptions
- **9 Bundled Voices** - Ready to use with full emotion support
- **Quick Speak** - Saved phrases for instant generation

## Changes Made for Portable Distribution

### 1. New Portable Path Management
- Created `src/myvoice/utils/portable_paths.py`
- All paths now resolve relative to `MyVoice.exe` location
- Works correctly in both frozen (executable) and development modes

### 2. Modified Files
- `src/myvoice/app.py` - Uses portable config paths
- `src/myvoice/main.py` - Uses portable log paths
- `src/myvoice/services/whisper_service.py` - Uses portable model cache

### 3. Directory Structure
```
MyVoice/                      # Application root (where MyVoice.exe lives)
├── MyVoice.exe              # Main executable
├── _internal/               # PyInstaller bundled files (DO NOT MODIFY)
├── config/                  # User settings (created on first run)
│   └── settings.json
├── logs/                    # Application logs
│   └── myvoice.log
├── voice_files/             # Voice samples and user voices
│   ├── embeddings/          # Saved voice embeddings (cloned/designed)
│   │   └── [voice_name]/    # Per-voice folder with embedding.pt, metadata.json
│   └── design_sessions/     # Voice Design working files
│       └── current/         # Current design session
├── whisper_models/          # Whisper AI models (downloaded/cached here)
├── qwen_models/             # Qwen3-TTS models (~3.4GB each, downloaded on first use)
└── README.txt               # User documentation
```

## Building the Portable Distribution

### Prerequisites
1. Python 3.10+ installed
2. All dependencies installed: `pip install -r requirements.txt`
3. PyInstaller installed: `pip install pyinstaller`

### Build Steps

1. **Navigate to build tools directory:**
   ```bash
   cd G:\MyVoicePublicInst\build_tools
   ```

2. **Run the portable build script:**
   ```bash
   python build_portable.py
   ```

3. **The script will:**
   - Clean previous builds
   - Run PyInstaller with `myvoice.spec`
   - Create portable directory structure
   - Copy default voice files
   - Generate README.txt for users

4. **Output location:**
   ```
   build_tools/dist/MyVoice/
   ```

### Build Output

After a successful build, you'll have:
```
build_tools/dist/MyVoice/
├── MyVoice.exe              ✓ Built by PyInstaller
├── _internal/               ✓ Python runtime + dependencies
├── config/                  ✓ Empty folder (for user settings)
├── logs/                    ✓ Empty folder (for logs)
├── voice_files/             ✓ Contains default voice samples
├── whisper_models/          ✓ Empty folder (models download on first use)
└── README.txt               ✓ User instructions
```

## Testing the Portable Build

### Test 1: Basic Functionality
1. Navigate to `build_tools/dist/MyVoice/`
2. Double-click `MyVoice.exe`
3. Verify the application starts
4. Check that `config/settings.json` is created
5. Check that `logs/myvoice.log` is created

### Test 2: Portability
1. Copy entire `MyVoice/` folder to a different location (e.g., Desktop)
2. Run `MyVoice.exe` from the new location
3. Verify settings are stored in the new location's `config/` folder
4. Verify logs are written to the new location's `logs/` folder

### Test 3: USB Drive
1. Copy `MyVoice/` folder to a USB drive
2. Run from USB drive
3. Verify all features work
4. Verify settings persist on the USB drive

### Test 4: Multiple Instances
1. Copy `MyVoice/` to two different locations
2. Run both instances simultaneously
3. Verify they use separate config files
4. Verify they don't interfere with each other

## Distribution

### Creating the Distribution Archive
```bash
# Navigate to build output
cd build_tools/dist

# Create ZIP archive (Windows)
powershell Compress-Archive -Path MyVoice -DestinationPath MyVoice-Portable-v1.0.zip

# Or use 7-Zip for better compression
7z a -tzip -mx=9 MyVoice-Portable-v1.0.zip MyVoice/
```

**Size:** ~250-400MB compressed

Note: Qwen3-TTS voice cloning is embedded in the application. Models (~3.4GB) are downloaded automatically on first use.

## Troubleshooting

### Build fails with "module not found"
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one

### Executable won't start
- Check for antivirus blocking
- Verify all DLLs are in `_internal/`
- Check `logs/myvoice.log` for errors

### Settings not saving
- Verify the application has write permissions
- Don't place in protected folders (e.g., `C:\Program Files`)
- Check folder isn't read-only

### Voice cloning not working
- Ensure sufficient disk space for model download (~3.5GB)
- Check internet connection for first-time model download
- Review logs for error details

## Advanced: Manual Build Process

If you prefer to build manually:

```bash
cd build_tools

# Clean previous builds
rm -rf build dist

# Run PyInstaller
python -m PyInstaller myvoice.spec --clean

# Create portable structure
mkdir -p dist/MyVoice/config
mkdir -p dist/MyVoice/logs
mkdir -p dist/MyVoice/voice_files
mkdir -p dist/MyVoice/whisper_models

# Copy default voices
cp -r ../src/install_files/default_voices/* dist/MyVoice/voice_files/
```

## Verification Checklist

Before distributing, verify:

### Core Functionality
- [ ] `MyVoice.exe` starts without errors
- [ ] Settings save to `config/settings.json`
- [ ] Logs write to `logs/myvoice.log`
- [ ] Voice files load from `voice_files/`
- [ ] Application works when moved to different folder
- [ ] No files created in `%APPDATA%` or `%LOCALAPPDATA%`
- [ ] README.txt contains correct instructions

### V2 Features
- [ ] Bundled voices (9 timbres) appear in voice selector
- [ ] Emotion buttons visible and functional
- [ ] Voice Design dialog works (preview + save)
- [ ] Voice Clone dialog works (extract + save)
- [ ] Quick Speak menu shows and triggers phrases
- [ ] Qwen3-TTS models download on first use
- [ ] Model switching works (bundled → cloned → designed)
- [ ] Saved voices appear in `voice_files/embeddings/`

## File Size Reference

| Component | Approximate Size |
|-----------|------------------|
| MyVoice.exe | ~24 MB |
| _internal/ | ~300 MB |
| voice_files/ (default samples) | ~30 MB |
| whisper_models/ (after first use) | ~150 MB |
| Qwen3-TTS models (one at a time) | ~3.4 GB each |
| **Total (distribution)** | **~350 MB** |
| **Total (with one TTS model + Whisper)** | **~3.9 GB** |
| **Total (with all 3 TTS models)** | **~10.5 GB** |

**Note:** V2 uses lazy loading - only one Qwen3-TTS model loads at a time:
- Base model: For cloned voices
- CustomVoice model: For bundled voices
- VoiceDesign model: For designed voices

## Support

For build issues:
1. Check `build/build.log` for PyInstaller errors
2. Review `dist/MyVoice/logs/myvoice.log` for runtime errors
3. Verify paths in `portable_paths.py` are correct
4. Test in development mode first: `python src/myvoice/main.py`

---

**Last Updated:** 2026-02-08
**Build Script Version:** 2.0
**Compatible with:** MyVoice v2.0.0
