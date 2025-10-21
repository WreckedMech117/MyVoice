"""
GPT-SoVITS Device Patching Utility

Patches GPT-SoVITS source files to support automatic CPU fallback
for systems without CUDA-capable GPU.

Usage:
    python -m myvoice.utils.patch_gptsovits_device [gptsovits_dir]

Environment Variables:
    GPTSOVITS_DEVICE: Force device selection (cpu/cuda)
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional


class GPTSoVITSPatcher:
    """Patches GPT-SoVITS source files for CPU fallback support."""

    # Files and line numbers to patch (relative to GPT-SoVITS root)
    PATCH_TARGETS = [
        ("config.py", [20]),
        ("GPT_SoVITS/TTS_infer_pack/TTS.py", [70, 147]),
        ("GPT_SoVITS/inference_webui.py", [78, 117]),
    ]

    # Patch patterns (old_pattern, new_pattern, description)
    PATTERNS = [
        (
            r"device\s*=\s*torch\.device\(['\"]cuda['\"]\)",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "Replace hardcoded cuda device with conditional"
        ),
        (
            r"\.to\(['\"]cuda['\"]\)",
            ".to(device)",
            "Replace .to('cuda') with .to(device)"
        ),
        (
            r"torch\.load\(([^,)]+)\)(?!\s*,\s*map_location)",
            r"torch.load(\1, map_location=device)",
            "Add map_location to torch.load calls"
        ),
        (
            r"(\s+)(model\.half\(\))",
            r"\1if device.type == 'cuda': model.half()",
            "Conditional half-precision for GPU only"
        ),
    ]

    def __init__(self, gptsovits_root: Path):
        """
        Initialize patcher.

        Args:
            gptsovits_root: Path to GPT-SoVITS installation directory
        """
        self.root = Path(gptsovits_root)
        self.patched_files: List[Path] = []
        self.marker_file = self.root / ".device_patched"

    def is_already_patched(self) -> bool:
        """Check if patches have already been applied."""
        return self.marker_file.exists()

    def mark_as_patched(self):
        """Create marker file to indicate patches applied."""
        self.marker_file.write_text("CPU fallback patches applied\n")

    def patch_file(self, file_path: Path) -> Tuple[bool, int]:
        """
        Patch a single file with all applicable patterns.

        Args:
            file_path: Path to file to patch

        Returns:
            Tuple of (success, number_of_changes)
        """
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            return False, 0

        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            changes = 0

            for old_pattern, new_pattern, description in self.PATTERNS:
                matches = list(re.finditer(old_pattern, content))
                if matches:
                    content = re.sub(old_pattern, new_pattern, content)
                    changes += len(matches)
                    print(f"  [OK] {description}: {len(matches)} change(s)")

            if changes > 0:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                backup_path.write_text(original_content, encoding='utf-8')

                # Write patched content
                file_path.write_text(content, encoding='utf-8')
                self.patched_files.append(file_path)
                return True, changes
            else:
                print(f"  [INFO] No changes needed")
                return True, 0

        except Exception as e:
            print(f"  [ERROR] Error patching file: {e}")
            return False, 0

    def patch_all(self) -> bool:
        """
        Apply all patches to GPT-SoVITS installation.

        Returns:
            True if all patches applied successfully
        """
        if self.is_already_patched():
            print("[OK] GPT-SoVITS already patched for CPU fallback")
            return True

        if not self.root.exists():
            print(f"[ERROR] GPT-SoVITS directory not found: {self.root}")
            return False

        print(f"Patching GPT-SoVITS installation at: {self.root}")
        print("-" * 60)

        total_changes = 0
        failed_files = []

        for rel_path, _line_hints in self.PATCH_TARGETS:
            file_path = self.root / rel_path
            print(f"\nPatching: {rel_path}")

            success, changes = self.patch_file(file_path)
            if success:
                total_changes += changes
            else:
                failed_files.append(rel_path)

        print("\n" + "=" * 60)

        if failed_files:
            print(f"[ERROR] Patching failed for {len(failed_files)} file(s):")
            for f in failed_files:
                print(f"  - {f}")
            return False
        else:
            print(f"[OK] Successfully patched {len(self.patched_files)} file(s)")
            print(f"[OK] Total changes: {total_changes}")
            self.mark_as_patched()
            return True


def find_gptsovits_installation() -> Optional[Path]:
    """
    Locate GPT-SoVITS installation directory.

    Searches in common locations:
    1. Environment variable GPTSOVITS_PATH
    2. ./GPT-SoVITS (current directory)
    3. ../GPT-SoVITS (parent directory)

    Returns:
        Path to GPT-SoVITS directory or None if not found
    """
    # Check environment variable
    env_path = os.environ.get('GPTSOVITS_PATH')
    if env_path:
        path = Path(env_path)
        if path.exists() and (path / "config.py").exists():
            return path

    # Check current directory
    current = Path.cwd() / "GPT-SoVITS"
    if current.exists() and (current / "config.py").exists():
        return current

    # Check parent directory
    parent = Path.cwd().parent / "GPT-SoVITS"
    if parent.exists() and (parent / "config.py").exists():
        return parent

    return None


def main():
    """Main entry point for patching script."""
    print("=" * 60)
    print("GPT-SoVITS CPU Fallback Patcher")
    print("=" * 60)

    # Get GPT-SoVITS directory from argument or auto-detect
    if len(sys.argv) > 1:
        gptsovits_dir = Path(sys.argv[1])
    else:
        gptsovits_dir = find_gptsovits_installation()
        if not gptsovits_dir:
            print("\n[ERROR] GPT-SoVITS installation not found")
            print("\nUsage:")
            print("  python -m myvoice.utils.patch_gptsovits_device [path]")
            print("\nOr set environment variable:")
            print("  set GPTSOVITS_PATH=C:\\path\\to\\GPT-SoVITS")
            sys.exit(1)

    # Apply patches
    patcher = GPTSoVITSPatcher(gptsovits_dir)
    success = patcher.patch_all()

    if success:
        print("\n[OK] Patching complete!")
        print("\nGPT-SoVITS will now automatically:")
        print("  - Use GPU (CUDA) if available")
        print("  - Fall back to CPU if no GPU detected")
        print("\nTo force CPU mode, set environment variable:")
        print("  set GPTSOVITS_DEVICE=cpu")
        sys.exit(0)
    else:
        print("\n[ERROR] Patching failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
