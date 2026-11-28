"""Stage the Lightning engine runtime (qwen3-tts.cpp DLLs + 4-bit GGUF models) into <repo>/lightning.

Usage: python build_tools/stage_lightning.py <qwen3-tts.cpp build tree> <model variant dir>
  e.g. python build_tools/stage_lightning.py I:\LightningSpike\qwen3-tts.cpp I:\LightningSpike\variants\q4k-embq8-tokq8

The staged folder is what the app loads in development (see
``myvoice.services.lightning.engine.lightning_dir``) and what ``myvoice.spec``
bundles. It is gitignored: ~1.2 GB of binaries. How those binaries are built is
recorded in ``_bmad-output/implementation-artifacts/19-1-lightning-build-recipe.md``.
The CUDA runtime DLLs are NOT staged: the engine reuses the copies torch ships.
"""
import shutil
import sys
from pathlib import Path

DLLS = {
    "build/qwen3tts.dll",
    "ggml/build/bin/ggml.dll",
    "ggml/build/bin/ggml-base.dll",
    "ggml/build/bin/ggml-cpu.dll",
    "ggml/build/bin/ggml-cuda.dll",
}
# Names are fixed by the qwen3-tts.cpp loader, whatever quantization they hold.
MODELS = ("qwen3-tts-0.6b-q8_0.gguf", "qwen3-tts-tokenizer-f16.gguf")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    tree, variant = Path(sys.argv[1]), Path(sys.argv[2])
    dest = Path(__file__).resolve().parent.parent / "lightning"
    (dest / "bin").mkdir(parents=True, exist_ok=True)
    (dest / "models").mkdir(parents=True, exist_ok=True)
    for rel in sorted(DLLS):
        src = tree / rel
        shutil.copy2(src, dest / "bin" / src.name)
        print(f"bin/{src.name}  {src.stat().st_size / 2**20:.1f} MB")
    for name in MODELS:
        src = variant / name
        shutil.copy2(src, dest / "models" / name)
        print(f"models/{name}  {src.stat().st_size / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
