"""
FFmpeg Setup Script for MyVoice
Downloads and extracts FFmpeg binaries for Windows
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
FFMPEG_DIR = Path(__file__).parent / "ffmpeg"
TEMP_DIR = Path(__file__).parent / "temp_ffmpeg"


def download_file(url, dest_path):
    """Download file with progress indication"""
    print(f"Downloading FFmpeg from {url}...")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\rProgress: {percent:.1f}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading FFmpeg: {e}")
        return False


def extract_ffmpeg(zip_path, extract_to):
    """Extract FFmpeg binaries from zip"""
    print("Extracting FFmpeg...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all files to temp directory
            zip_ref.extractall(extract_to)

        # Find the bin directory in extracted files
        for root, dirs, files in os.walk(extract_to):
            if 'bin' in dirs:
                bin_dir = Path(root) / 'bin'

                # Create ffmpeg directory if it doesn't exist
                FFMPEG_DIR.mkdir(exist_ok=True)

                # Copy ffmpeg.exe and ffprobe.exe
                for exe in ['ffmpeg.exe', 'ffprobe.exe']:
                    src = bin_dir / exe
                    if src.exists():
                        dst = FFMPEG_DIR / exe
                        shutil.copy2(src, dst)
                        print(f"Installed: {exe}")

                return True

        print("Error: Could not find FFmpeg binaries in archive")
        return False

    except Exception as e:
        print(f"Error extracting FFmpeg: {e}")
        return False


def cleanup(temp_dir, zip_path):
    """Remove temporary files"""
    print("Cleaning up temporary files...")

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if zip_path.exists():
            os.remove(zip_path)
        print("Cleanup complete!")
    except Exception as e:
        print(f"Warning: Could not clean up some temporary files: {e}")


def verify_installation():
    """Verify FFmpeg was installed correctly"""
    ffmpeg_exe = FFMPEG_DIR / "ffmpeg.exe"
    ffprobe_exe = FFMPEG_DIR / "ffprobe.exe"

    if ffmpeg_exe.exists() and ffprobe_exe.exists():
        print(f"\n✓ FFmpeg successfully installed to: {FFMPEG_DIR}")
        return True
    else:
        print(f"\n✗ FFmpeg installation failed")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("MyVoice - FFmpeg Setup")
    print("=" * 60)

    # Check if FFmpeg already exists
    if (FFMPEG_DIR / "ffmpeg.exe").exists() and (FFMPEG_DIR / "ffprobe.exe").exists():
        print("\nFFmpeg is already installed!")
        print(f"Location: {FFMPEG_DIR}")

        response = input("\nDo you want to reinstall? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return 0

        # Remove existing installation
        print("Removing existing installation...")
        shutil.rmtree(FFMPEG_DIR, ignore_errors=True)

    # Create temp directory
    TEMP_DIR.mkdir(exist_ok=True)
    zip_path = TEMP_DIR / "ffmpeg.zip"

    try:
        # Download FFmpeg
        if not download_file(FFMPEG_URL, zip_path):
            return 1

        # Extract FFmpeg
        if not extract_ffmpeg(zip_path, TEMP_DIR):
            return 1

        # Verify installation
        if not verify_installation():
            return 1

        print("\n" + "=" * 60)
        print("Setup complete! FFmpeg is ready to use.")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        return 1

    finally:
        # Always cleanup temp files
        cleanup(TEMP_DIR, zip_path)


if __name__ == "__main__":
    sys.exit(main())
