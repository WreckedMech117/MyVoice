"""
Portable Build Script for MyVoice

This script builds a fully portable distribution of MyVoice that can be:
- Zipped and distributed as a single archive
- Extracted and run without installation
- Moved to any location without breaking

All user data, settings, and voice files are stored locally
within the application folder, not in %APPDATA% or other system directories.

Usage:
    python build_portable.py

Output:
    dist/MyVoice/  - Ready to zip and distribute
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def main():
    """Build portable distribution of MyVoice."""
    print("=" * 70)
    print("MyVoice Portable Build Script")
    print("=" * 70)
    print()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    spec_file = script_dir / "myvoice.spec"
    dist_dir = script_dir / "dist" / "MyVoice"

    print(f"Project root: {project_root}")
    print(f"Spec file: {spec_file}")
    print(f"Output directory: {dist_dir}")
    print()

    # Step 1: Clean previous builds
    print("[1/5] Cleaning previous builds...")
    build_dir = script_dir / "build"
    if build_dir.exists():
        print(f"  Removing {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)

    if dist_dir.exists():
        print(f"  Removing {dist_dir}")
        shutil.rmtree(dist_dir, ignore_errors=True)

    print("  ✓ Clean complete")
    print()

    # Step 2: Run PyInstaller
    print("[2/5] Running PyInstaller to build executable...")
    if not spec_file.exists():
        print(f"  ERROR: Spec file not found at {spec_file}")
        return 1

    # Use portable Python 3.10 if available
    python_exe = project_root / "python310" / "python.exe"
    if not python_exe.exists():
        # Fallback to system Python
        python_exe = Path(sys.executable)

    print(f"  Using Python: {python_exe}")

    try:
        result = subprocess.run(
            [str(python_exe), "-m", "PyInstaller", str(spec_file), "--clean"],
            cwd=str(script_dir),
            check=True,
            capture_output=True,
            text=True
        )
        print("  ✓ PyInstaller build complete")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: PyInstaller failed:")
        print(e.stdout)
        print(e.stderr)
        return 1
    print()

    # Step 3: Verify build output
    print("[3/5] Verifying build output...")
    exe_path = dist_dir / "MyVoice.exe"
    if not exe_path.exists():
        print(f"  ERROR: Executable not found at {exe_path}")
        return 1
    print(f"  ✓ Found MyVoice.exe ({exe_path.stat().st_size // 1024 // 1024} MB)")
    print()

    # Step 4: Create portable directory structure
    print("[4/5] Creating portable directory structure...")

    # Create directories
    directories = [
        dist_dir / "config",
        dist_dir / "logs",
        dist_dir / "voice_files",
        dist_dir / "whisper_models",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {directory.relative_to(dist_dir)}/")

    print()

    # Step 5: Copy default voice files
    print("[5/5] Copying default voice files...")
    default_voices_source = project_root / "src" / "install_files" / "default_voices"
    if default_voices_source.exists():
        for voice_file in default_voices_source.rglob("*"):
            if voice_file.is_file():
                relative_path = voice_file.relative_to(default_voices_source)
                dest_path = dist_dir / "voice_files" / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(voice_file, dest_path)
                print(f"  ✓ Copied {relative_path}")
    else:
        print(f"  ⚠ Default voices not found at {default_voices_source}")
    print()

    # Create README
    readme_content = """# MyVoice - Portable Distribution

## Quick Start

1. Extract this ZIP file to any location (e.g., C:\\MyVoice or D:\\Apps\\MyVoice)
2. Run MyVoice.exe
3. All settings, logs, and voice files will be stored in this folder

## Folder Structure

```
MyVoice/
├── MyVoice.exe          # Main application
├── _internal/           # Python runtime and dependencies (do not modify)
├── config/              # Settings stored here (created on first run)
├── logs/                # Application logs
├── voice_files/         # Your voice samples
└── whisper_models/      # AI models for transcription
```

## Portable Features

✓ **No installation required** - Just extract and run
✓ **Self-contained** - All data stored locally in this folder
✓ **Movable** - Copy entire folder to USB drive or different PC
✓ **No registry changes** - Clean uninstall by deleting folder
✓ **Multi-instance safe** - Run from different folders simultaneously

## Voice Cloning with Qwen3-TTS

MyVoice uses embedded Qwen3-TTS for voice cloning. Models are downloaded automatically
on first use. You can choose between two model tiers in Settings > TTS Settings:

- **Quality (1.7B)**: Higher output quality (~3.4 GB VRAM, ~3.4 GB download)
- **Small (0.6B)**: Lower VRAM requirements (~1.2 GB VRAM, ~1.2 GB download)

Voice cloning features work fully offline after initial model download.

## Requirements

- Windows 10 or later (64-bit)
- ~500MB disk space (application)
- ~1.2-3.5GB additional space (Qwen3-TTS models, depending on quality tier selected)

## Troubleshooting

**MyVoice.exe won't start:**
- Check Windows Defender/antivirus isn't blocking it
- Ensure you have sufficient disk space
- Check logs/myvoice.log for error details

**Settings not saving:**
- Ensure the folder is not in a protected location (e.g., Program Files)
- Check folder permissions allow write access

**Voice cloning not working:**
- Ensure sufficient disk space for model download (~3.5GB)
- Check internet connection for first-time model download
- Review logs for error details

## Support

For issues and updates, visit: https://github.com/myvoice/myvoice

---

MyVoice Portable Distribution - Built with Qwen3-TTS Voice Cloning
"""

    readme_path = dist_dir / "README.txt"
    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"✓ Created README.txt")
    print()

    # Final summary
    print()
    print("=" * 70)
    print("BUILD COMPLETE!")
    print("=" * 70)
    print()
    print(f"✓ Portable distribution ready at: {dist_dir}")
    print()

    # Calculate sizes
    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
    total_mb = total_size // 1024 // 1024

    print("Directory size breakdown:")
    print(f"  Total distribution size: {total_mb} MB")
    print()
    print("✓ PORTABLE DISTRIBUTION: Includes embedded Qwen3-TTS voice cloning")
    print("  Voice cloning models (~3.4GB) download automatically on first use")
    print()

    print("To distribute:")
    print(f"  1. Compress the folder (recommended: 7-Zip for better compression)")
    print(f"     7z a -tzip -mx=9 MyVoice-Portable-v1.0.zip MyVoice\\")
    print()
    print(f"  2. Upload to distribution platform")
    print()
    print(f"  3. Users simply:")
    print(f"     - Extract MyVoice-Portable-v1.0.zip")
    print(f"     - Run MyVoice.exe")
    print(f"     - All features ready to use!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
