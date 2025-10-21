"""
MyVoice User Data Migration Utility

This script migrates user data from the installation directory to %LOCALAPPDATA%
following Windows best practices. This allows the application to run without
write permissions in Program Files.

Migration includes:
- Configuration files (config/settings.json)
- Log files (logs/*.log)
- Voice files (voice_files/*)
- Backup files (config/*.bak*)

Usage:
    python -m myvoice.utils.migrate_user_data
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Tuple, List


def get_paths() -> Tuple[Path, Path]:
    """
    Get source (installation) and destination (%LOCALAPPDATA%) paths.

    Returns:
        Tuple of (source_dir, dest_dir)
    """
    # Source: Installation directory (current working directory)
    source_dir = Path.cwd()

    # Destination: %LOCALAPPDATA%\MyVoice on Windows, ~/.local/share/MyVoice on Unix
    if sys.platform == "win32":
        appdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        dest_dir = Path(appdata) / "MyVoice"
    else:
        dest_dir = Path.home() / ".local" / "share" / "MyVoice"

    return source_dir, dest_dir


def migrate_directory(source: Path, dest: Path, dir_name: str, file_patterns: List[str] = None) -> Tuple[int, int]:
    """
    Migrate files from source directory to destination directory.

    Args:
        source: Source base directory
        dest: Destination base directory
        dir_name: Directory name to migrate (e.g., "config", "logs")
        file_patterns: Optional list of file patterns to copy (default: all files)

    Returns:
        Tuple of (files_copied, errors)
    """
    source_subdir = source / dir_name
    dest_subdir = dest / dir_name

    if not source_subdir.exists():
        print(f"[INFO] Source directory not found: {source_subdir}")
        return 0, 0

    # Create destination directory
    dest_subdir.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    errors = 0

    # Get all files in source directory
    if file_patterns:
        files_to_copy = []
        for pattern in file_patterns:
            files_to_copy.extend(source_subdir.glob(pattern))
    else:
        files_to_copy = list(source_subdir.rglob('*'))

    for source_file in files_to_copy:
        if not source_file.is_file():
            continue

        # Calculate relative path and destination
        rel_path = source_file.relative_to(source_subdir)
        dest_file = dest_subdir / rel_path

        # Skip if destination already exists and is newer
        if dest_file.exists():
            if dest_file.stat().st_mtime > source_file.stat().st_mtime:
                print(f"[SKIP] Destination is newer: {rel_path}")
                continue

        try:
            # Create parent directories if needed
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(source_file, dest_file)
            files_copied += 1
            print(f"[OK] Copied: {dir_name}/{rel_path}")
        except Exception as e:
            errors += 1
            print(f"[ERROR] Failed to copy {rel_path}: {e}")

    return files_copied, errors


def main():
    """Main migration function."""
    print("========================================")
    print("MyVoice User Data Migration Utility")
    print("========================================")
    print()

    # Get source and destination paths
    source_dir, dest_dir = get_paths()

    print(f"Source (Installation): {source_dir}")
    print(f"Destination (AppData): {dest_dir}")
    print()

    # Check if source directories exist
    has_data = False
    for dir_name in ["config", "logs", "voice_files"]:
        if (source_dir / dir_name).exists():
            has_data = True
            break

    if not has_data:
        print("[INFO] No user data found in installation directory.")
        print("[INFO] Migration not needed - application will create new directories.")
        print()
        return 0

    # Confirm with user
    print("This will copy your user data to the new location.")
    print("Your original files will not be deleted.")
    print()
    response = input("Proceed with migration? (y/n): ").strip().lower()

    if response != 'y':
        print("[CANCELLED] Migration cancelled by user.")
        return 1

    print()
    print("Starting migration...")
    print()

    total_copied = 0
    total_errors = 0

    # Migrate configuration files
    print("[1/3] Migrating configuration files...")
    copied, errors = migrate_directory(
        source_dir, dest_dir, "config",
        file_patterns=["*.json", "*.bak*"]
    )
    total_copied += copied
    total_errors += errors
    print(f"       Copied: {copied}, Errors: {errors}")
    print()

    # Migrate log files
    print("[2/3] Migrating log files...")
    copied, errors = migrate_directory(
        source_dir, dest_dir, "logs",
        file_patterns=["*.log"]
    )
    total_copied += copied
    total_errors += errors
    print(f"       Copied: {copied}, Errors: {errors}")
    print()

    # Migrate voice files
    print("[3/3] Migrating voice files...")
    copied, errors = migrate_directory(
        source_dir, dest_dir, "voice_files"
    )
    total_copied += copied
    total_errors += errors
    print(f"       Copied: {copied}, Errors: {errors}")
    print()

    # Summary
    print("========================================")
    print("Migration Summary")
    print("========================================")
    print(f"Total files copied: {total_copied}")
    print(f"Total errors: {total_errors}")
    print()

    if total_errors == 0:
        print("[SUCCESS] Migration completed successfully!")
        print()
        print("Your user data is now in:")
        print(f"  {dest_dir}")
        print()
        print("The application will now use this location for:")
        print("  - Configuration: %LOCALAPPDATA%\\MyVoice\\config")
        print("  - Logs: %LOCALAPPDATA%\\MyVoice\\logs")
        print("  - Voice Files: %LOCALAPPDATA%\\MyVoice\\voice_files")
        print()
        print("You can safely delete the old folders from the installation")
        print("directory after verifying everything works.")
    else:
        print("[WARNING] Migration completed with errors.")
        print("Please review the error messages above.")

    print()
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print()
        print("[CANCELLED] Migration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"[FATAL ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
