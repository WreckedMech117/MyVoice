#!/usr/bin/env python3
"""
MyVoice Version Management System

Central version control for consistent versioning across:
- Application (__init__.py)
- PyInstaller spec file
- Inno Setup installer script
- Release artifacts

Usage:
    python version.py                    # Display current version
    python version.py bump major         # 1.0.0 -> 2.0.0
    python version.py bump minor         # 1.0.0 -> 1.1.0
    python version.py bump patch         # 1.0.0 -> 1.0.1
    python version.py set 1.2.3          # Set specific version
    python version.py update-all         # Update all files with current version
"""

import re
import sys
from pathlib import Path
from typing import Tuple

# =============================================================================
# VERSION CONFIGURATION
# =============================================================================

# Single source of truth for version number
VERSION_MAJOR = 2
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_BUILD = 22  # Auto-incremented on each build

# Derived version strings
VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
VERSION_FULL = f"{VERSION}.{VERSION_BUILD}"

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
BUILD_TOOLS = PROJECT_ROOT / "build_tools"
SRC_PATH = PROJECT_ROOT / "src"

FILES_TO_UPDATE = {
    "app_init": SRC_PATH / "myvoice" / "__init__.py",
    "spec_file": BUILD_TOOLS / "myvoice.spec",
    "installer": BUILD_TOOLS / "installer.iss",
}

# =============================================================================
# VERSION PARSING AND FORMATTING
# =============================================================================

def parse_version(version_string: str) -> Tuple[int, int, int]:
    """Parse version string into (major, minor, patch) tuple."""
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_string)
    if not match:
        raise ValueError(f"Invalid version format: {version_string}")
    return tuple(map(int, match.groups()))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version tuple into string."""
    return f"{major}.{minor}.{patch}"


# =============================================================================
# VERSION OPERATIONS
# =============================================================================

def bump_version(component: str) -> Tuple[int, int, int]:
    """
    Bump version component.

    Args:
        component: 'major', 'minor', or 'patch'

    Returns:
        New version tuple (major, minor, patch)
    """
    major, minor, patch = VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH

    if component == 'major':
        major += 1
        minor = 0
        patch = 0
    elif component == 'minor':
        minor += 1
        patch = 0
    elif component == 'patch':
        patch += 1
    else:
        raise ValueError(f"Invalid component: {component}. Use: major, minor, or patch")

    return major, minor, patch


def set_version(version_string: str) -> Tuple[int, int, int]:
    """Set specific version from string."""
    return parse_version(version_string)


# =============================================================================
# FILE UPDATES
# =============================================================================

def update_app_init(major: int, minor: int, patch: int) -> bool:
    """Update version in src/myvoice/__init__.py"""
    file_path = FILES_TO_UPDATE["app_init"]

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return False

    content = file_path.read_text(encoding='utf-8')
    new_version = format_version(major, minor, patch)

    # Update __version__ = "x.y.z"
    pattern = r'__version__\s*=\s*["\'][\d.]+["\']'
    replacement = f'__version__ = "{new_version}"'

    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        file_path.write_text(new_content, encoding='utf-8')
        print(f"+ Updated {file_path.name}: __version__ = \"{new_version}\"")
        return True
    elif re.search(pattern, content) is not None:
        print(f"= {file_path.name}: __version__ already at {new_version}")
        return True  # Already at target; not a failure
    else:
        print(f"! No __version__ assignment found in {file_path.name}")
        return False


def update_spec_file(major: int, minor: int, patch: int) -> bool:
    """Update version in myvoice.spec (comments only, not used in build).

    Targets only the trailing docstring's "MyVoice-Portable-v<version>.zip"
    illustrative example to avoid damaging unrelated MyVoice-prefixed text
    elsewhere in the spec. The spec file itself does not consume the
    version string at build time — this update is purely documentary.
    """
    file_path = FILES_TO_UPDATE["spec_file"]

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return False

    content = file_path.read_text(encoding='utf-8')
    new_version = format_version(major, minor, patch)

    # Targeted pattern: only the docstring's "MyVoice-Portable-v<version>.zip"
    # illustrative example. Anchored to the literal "Portable-v" prefix
    # and the literal ".zip" suffix so we don't damage other MyVoice
    # references (e.g., "MyVoice PyInstaller Specification File").
    # Tolerates 1.0, 2.1.0, or any dotted-numeric version present in the
    # docstring; rewrites to the canonical 3-component format.
    pattern = r'(MyVoice-Portable-v)\d+(?:\.\d+)*(\.zip)'
    replacement = rf'\g<1>{new_version}\g<2>'

    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        file_path.write_text(new_content, encoding='utf-8')
        print(f"+ Updated {file_path.name}: docstring example version")
        return True
    elif re.search(pattern, content) is not None:
        print(f"= {file_path.name}: docstring example already at {new_version}")
        return True  # Already at target; not a failure
    else:
        print(f"! No 'MyVoice-Portable-vX.Y.Z.zip' pattern found in {file_path.name}")
        return False


def update_installer_script(major: int, minor: int, patch: int) -> bool:
    """Update version + build number in installer.iss.

    Synchronizes both ``#define MyAppVersion "x.y.z"`` (consumed by
    Add/Remove Programs entry, registry Version key, OutputBaseFilename)
    and ``#define MyAppBuild "N"`` (consumed by registry Build key for
    log-level traceability per Story tooling-2 AC #5).
    """
    file_path = FILES_TO_UPDATE["installer"]

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return False

    content = file_path.read_text(encoding='utf-8')
    new_version = format_version(major, minor, patch)

    # Update #define MyAppVersion "x.y.z"
    new_content = re.sub(
        r'(#define\s+MyAppVersion\s+")[^"]+(")',
        rf'\g<1>{new_version}\g<2>',
        content,
    )

    # Update #define MyAppBuild "N" (Story tooling-2 AC #5).
    # Sourced from VERSION_BUILD which is re-read from disk on each Python
    # invocation, so a prior `version.py increment-build` is reflected here.
    new_content = re.sub(
        r'(#define\s+MyAppBuild\s+")[^"]+(")',
        rf'\g<1>{VERSION_BUILD}\g<2>',
        new_content,
    )

    if new_content != content:
        file_path.write_text(new_content, encoding='utf-8')
        print(f"+ Updated {file_path.name}: MyAppVersion=\"{new_version}\", MyAppBuild=\"{VERSION_BUILD}\"")
        return True
    elif re.search(r'#define\s+MyAppVersion\s+"', content) is not None:
        print(f"= {file_path.name}: MyAppVersion already at {new_version}, MyAppBuild already at {VERSION_BUILD}")
        return True  # Already at target; not a failure
    else:
        print(f"! No MyAppVersion/MyAppBuild defines found in {file_path.name}")
        return False


def update_version_py(major: int, minor: int, patch: int) -> bool:
    """Update version constants in this file."""
    file_path = Path(__file__)
    content = file_path.read_text(encoding='utf-8')

    # Update VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH
    new_content = content
    new_content = re.sub(r'VERSION_MAJOR\s*=\s*\d+', f'VERSION_MAJOR = {major}', new_content)
    new_content = re.sub(r'VERSION_MINOR\s*=\s*\d+', f'VERSION_MINOR = {minor}', new_content)
    new_content = re.sub(r'VERSION_PATCH\s*=\s*\d+', f'VERSION_PATCH = {patch}', new_content)

    if new_content != content:
        file_path.write_text(new_content, encoding='utf-8')
        print(f"+ Updated {file_path.name}: VERSION constants")
        return True
    else:
        print(f"= {file_path.name}: VERSION constants already at {format_version(major, minor, patch)}")
        return True  # Already at target; not a failure


def increment_build_number() -> int:
    """Increment build number and update this file."""
    file_path = Path(__file__)
    content = file_path.read_text(encoding='utf-8')

    # Find current build number
    match = re.search(r'VERSION_BUILD\s*=\s*(\d+)', content)
    if not match:
        print("! VERSION_BUILD not found")
        return VERSION_BUILD

    current_build = int(match.group(1))
    new_build = current_build + 1

    # Update build number
    new_content = re.sub(
        r'VERSION_BUILD\s*=\s*\d+',
        f'VERSION_BUILD = {new_build}',
        content
    )

    file_path.write_text(new_content, encoding='utf-8')
    print(f"+ Build number incremented: {current_build} -> {new_build}")
    return new_build


def update_all_files(major: int, minor: int, patch: int):
    """Update version in all relevant files."""
    print(f"\nUpdating version to {format_version(major, minor, patch)}...\n")

    results = []
    results.append(update_version_py(major, minor, patch))
    results.append(update_app_init(major, minor, patch))
    results.append(update_spec_file(major, minor, patch))
    results.append(update_installer_script(major, minor, patch))

    print(f"\n{'='*60}")
    success_count = sum(results)
    print(f"Updated {success_count}/{len(results)} files successfully")
    print(f"{'='*60}\n")

    return all(results)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def display_version():
    """Display current version information."""
    print(f"""
MyVoice Version Information
{'='*60}
Version:        {VERSION}
Full Version:   {VERSION_FULL}
Major:          {VERSION_MAJOR}
Minor:          {VERSION_MINOR}
Patch:          {VERSION_PATCH}
Build:          {VERSION_BUILD}
{'='*60}
    """.strip())


def main():
    """Main CLI entry point."""
    if len(sys.argv) == 1:
        # No arguments: display current version
        display_version()
        return 0

    command = sys.argv[1].lower()

    try:
        if command == 'bump':
            if len(sys.argv) < 3:
                print("Error: Specify component to bump: major, minor, or patch")
                print("Usage: python version.py bump <major|minor|patch>")
                return 1

            component = sys.argv[2].lower()
            major, minor, patch = bump_version(component)
            update_all_files(major, minor, patch)
            print(f"\n+ Version bumped to {format_version(major, minor, patch)}")

        elif command == 'set':
            if len(sys.argv) < 3:
                print("Error: Specify version to set (e.g., 1.2.3)")
                print("Usage: python version.py set <version>")
                return 1

            version_string = sys.argv[2]
            major, minor, patch = set_version(version_string)
            update_all_files(major, minor, patch)
            print(f"\n+ Version set to {format_version(major, minor, patch)}")

        elif command == 'update-all':
            update_all_files(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)
            print(f"\n+ All files updated to {VERSION}")

        elif command == 'increment-build':
            new_build = increment_build_number()
            print(f"\n+ Build number is now {new_build}")
            print(f"Full version: {VERSION}.{new_build}")

        elif command in ['help', '--help', '-h']:
            print(__doc__)

        else:
            print(f"Error: Unknown command '{command}'")
            print(__doc__)
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
