# -*- mode: python ; coding: utf-8 -*-
"""
MyVoice PyInstaller Specification File
Builds standalone Windows executable with splash screen and icon

Build Command:
    pyinstaller myvoice.spec

Output:
    dist/MyVoice/MyVoice.exe
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project paths
# Use SPECPATH provided by PyInstaller (directory containing this spec file)
spec_dir = Path(SPECPATH) if 'SPECPATH' in dir() else Path(__file__).parent.resolve()
project_root = spec_dir.parent if spec_dir.name == 'build_tools' else spec_dir
src_path = project_root / 'src'
icon_path = src_path / 'icon'
myvoice_path = src_path / 'myvoice'

# =============================================================================
# HIDDEN IMPORTS
# =============================================================================

# PyQt6 - GUI framework
hiddenimports_pyqt6 = collect_submodules('PyQt6')

# PyTorch - Deep learning framework
# Note: Cannot use collect_all due to DLL loading issues during build
# Instead, we manually specify submodules and copy DLLs via runtime hook
hiddenimports_torch = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torch.version',
    'torch.backends',
    'torch.backends.cudnn',
    'torch.cuda',
    'torch.jit',
    'torch.autograd',
    'torch.distributed',
    'torch._C',
    'torchvision',
    'torchaudio',
    # torch._dynamo.polyfills modules - required by transformers
    'torch._dynamo',
    'torch._dynamo.polyfills',
    'torch._dynamo.polyfills.builtins',
    'torch._dynamo.polyfills.functools',
    'torch._dynamo.polyfills.fx',
    'torch._dynamo.polyfills.heapq',
    'torch._dynamo.polyfills.itertools',
    'torch._dynamo.polyfills.loader',
    'torch._dynamo.polyfills.operator',
    'torch._dynamo.polyfills.os',
    'torch._dynamo.polyfills.pytree',
    'torch._dynamo.polyfills.struct',
    'torch._dynamo.polyfills.sys',
    'torch._dynamo.polyfills.tensor',
    'torch._dynamo.polyfills._collections',
]
# Story 18.4 / D-22 Branch B — torch.compile lazy-imports the inductor
# backend's pattern-matching tables via importlib.import_module(name_string)
# at first-compilation time. PyInstaller's static analysis can't detect
# importlib-by-string, so the inductor's fx_passes.serialized_patterns
# subtree (and surrounding lazy-imports across _inductor + _functorch)
# must be force-collected for the bundle. Observed failure mode without
# this collection: at first user generation, the code_predictor's
# torch.compile-wrapped forward raises
# `ModuleNotFoundError: No module named 'torch._inductor.fx_passes.serialized_patterns'`,
# which propagates as BackendCompilerFailed → TRUE_STREAM emits 0 chunks
# → ValueError finalize() (Story 18.4 bundled-smoke, 2026-05-10 log).
hiddenimports_torch += collect_submodules('torch._inductor')
hiddenimports_torch += collect_submodules('torch._functorch')

# Story 18.4 fix #3 (2026-05-10) — collect_submodules('torch._inductor')
# did NOT recurse into `fx_passes/serialized_patterns/` (PyInstaller's
# package discovery skips this subdirectory; observed in the post-fix-#2
# bundle where `_internal/torch/_inductor/fx_passes/` was present but
# the `serialized_patterns/` subtree was missing entirely). Force-copy
# the directory as data files + enumerate the modules explicitly.
import glob as _glob  # used by fix #3 below + by the torch DLL block further down
_serialized_patterns_dir = (
    project_root / 'python310' / 'Lib' / 'site-packages' / 'torch'
    / '_inductor' / 'fx_passes' / 'serialized_patterns'
)
torch_serialized_patterns_datas = []
torch_serialized_patterns_hiddenimports = []
if _serialized_patterns_dir.exists():
    for _py in _glob.glob(str(_serialized_patterns_dir / '*.py')):
        _name = Path(_py).stem
        # Skip __pycache__ entries; pyc files only.
        if _name.startswith('__pycache__'):
            continue
        torch_serialized_patterns_datas.append(
            (_py, 'torch/_inductor/fx_passes/serialized_patterns')
        )
        if _name == '__init__':
            torch_serialized_patterns_hiddenimports.append(
                'torch._inductor.fx_passes.serialized_patterns'
            )
        else:
            torch_serialized_patterns_hiddenimports.append(
                f'torch._inductor.fx_passes.serialized_patterns.{_name}'
            )
    print(
        f"[SPEC] Force-copying {len(torch_serialized_patterns_datas)} "
        f"torch/_inductor/fx_passes/serialized_patterns/ files "
        f"(Story 18.4 torch.compile lazy-import fix)"
    )
hiddenimports_torch += torch_serialized_patterns_hiddenimports

# Torch DLL binaries - collected manually to avoid import issues
# (_glob already imported above for fix #3)
torch_binaries = []
_torch_lib = project_root / 'python310' / 'Lib' / 'site-packages' / 'torch' / 'lib'
if _torch_lib.exists():
    for _dll in _glob.glob(str(_torch_lib / '*.dll')):
        torch_binaries.append((_dll, 'torch/lib'))
        print(f"[SPEC] Adding torch DLL: {Path(_dll).name}")
torch_datas = []

# =============================================================================
# Story 18.5 (Phase ⊥-Polish-2-Ship) — triton-windows + Python 3.10.11 dev
# headers/libs + CUDA Toolkit redistributable subset bundling for the
# bundle-reach gate that closes Story 18.4's torch.compile machinery on
# production user machines.
#
# Architecture reference: `_bmad-output/planning-artifacts/
#   architecture-streaming-acceleration-and-lightning-tier.md` D-22 + D-23
#   (sealed 2026-05-10; LIVE in source-tree since Story 18.4; this story
#   makes them user-reachable in the bundled exe).
#
# License discipline: the bundled CUDA Toolkit subset ships ONLY files
# explicitly authorized by NVIDIA CUDA Toolkit EULA Attachment A (CUDA
# Runtime + NVRTC + nvrtc-builtins DLLs + device-side headers from crt/).
# NVCC is NOT bundled (EULA §1.1.2 #4 — developer tools, internal-use-only).
# Source-of-truth = `_bmad-output/implementation-artifacts/
#   18-5-cuda-toolkit-triton-bundling-evidence.md §"NVIDIA license clearance"`
# (gated on Commander sign-off per Story 18.5 Task 1.8).
# =============================================================================

# -----------------------------------------------------------------------------
# Block A — triton-windows hidden imports + datas
# -----------------------------------------------------------------------------
# triton's codegen pipeline uses importlib.import_module(name_string) at
# compile-time to load backend tables — same pattern as torch._inductor's
# fx_passes/serialized_patterns/ (the Story 18.4 Fix #3 precedent at
# `:83-:121`). PyInstaller's static analysis cannot detect importlib-by-
# string, so force-collect via collect_submodules() + collect_data_files().
#
# Without this block the bundled exe raises:
#   `RuntimeError: Cannot find a working triton installation`
# at first user TTS generation (the Story 18.4 Fix #4 failure class).
hiddenimports_triton = collect_submodules('triton')
triton_datas = collect_data_files('triton')
print(f"[SPEC] Story 18.5 — collected {len(hiddenimports_triton)} triton hidden imports + "
      f"{len(triton_datas)} triton data files")

# -----------------------------------------------------------------------------
# Block B — Python 3.10.11 dev headers + libs
# -----------------------------------------------------------------------------
# Required by triton's runtime kernel compilation pipeline. The portable
# embeddable-zip Python distribution that the production bundle is built
# from does NOT ship these by default. The build-host's `python310/Include/`
# + `python310/libs/python310.lib` are copied from a full python.org install
# at C:\Python310-fullinstall\ (one-time per build host; verified by the
# `[Bundle Prerequisites]` probes in build_release.bat).
#
# Story 18.5 Task 7 Fix #2 (2026-05-11): bundle paths corrected.
# triton-windows's `find_python()` at `triton/windows_utils.py:289-303`
# searches THREE locations for `libs/python{ver}.lib`:
#   - `sys.exec_prefix / libs / python310.lib`       (== _internal/libs/python310.lib in PyInstaller)
#   - `sys.base_exec_prefix / libs / python310.lib`  (== _internal/libs/python310.lib)
#   - `os.path.dirname(sys.executable) / libs / python310.lib`  (== dist/MyVoice/libs/python310.lib)
# So the lib MUST land at `_internal/libs/python310.lib`, NOT
# `_internal/python310/libs/python310.lib` (the V1 spec target).
# Similarly, `sysconfig.get_paths()["include"]` resolves to `_internal/Include/`
# in PyInstaller's frozen sysconfig — headers must land at `_internal/Include/`,
# NOT `_internal/python310/Include/`. Without these path fixes, triton's
# host-C-compile step fails with "Failed to find Python libs" warning and
# subsequent link errors.
python_headers_dir = project_root / 'python310' / 'Include'
python_libs_dir = project_root / 'python310' / 'libs'
python_headers_datas = []
if python_headers_dir.exists():
    for _h in _glob.glob(str(python_headers_dir / '**' / '*'), recursive=True):
        if Path(_h).is_file():
            _rel_parent = Path(_h).parent.relative_to(python_headers_dir)
            _dst = 'Include'  # Bundle at _internal/Include/ (PyInstaller sysconfig target)
            if str(_rel_parent) not in ('.', ''):
                _dst = f"{_dst}/{_rel_parent.as_posix()}"
            python_headers_datas.append((_h, _dst))
    print(f"[SPEC] Story 18.5 — bundling {len(python_headers_datas)} Python "
          f"dev header files from {python_headers_dir} -> _internal/Include/")
else:
    print(f"[SPEC] WARNING: Python dev headers missing at {python_headers_dir} — "
          f"triton's runtime kernel compilation will fail in the bundled exe; "
          f"see Story 18.5 Task 1.4 + the build_release.bat [Bundle Prerequisites] probe")

python_libs_datas = []
if (python_libs_dir / 'python310.lib').exists():
    # Bundle at _internal/libs/python310.lib (sys.exec_prefix-relative; the
    # path triton-windows's `find_python()` searches first).
    python_libs_datas.append(
        (str(python_libs_dir / 'python310.lib'), 'libs')
    )
    print(f"[SPEC] Story 18.5 — bundling python310.lib -> _internal/libs/")
else:
    print(f"[SPEC] WARNING: python310.lib missing at {python_libs_dir} — "
          f"triton's NVRTC build will fail in the bundled exe")

# -----------------------------------------------------------------------------
# Block C — CUDA Toolkit redistributable subset (NVIDIA EULA Attachment A scope)
# -----------------------------------------------------------------------------
# Bundle path = `_internal/cuda_redist/`. The directory name makes the
# redistribution-only scope explicit in `dir` listings (vs `cuda/` which
# would imply a full Toolkit install).
#
# Source = `build_tools/cuda_toolkit_subset/` produced by Story 18.5 Task 2.1's
# `stage_cuda_subset.py` (committed staging script). The script is the
# canonical recipe; its output is gitignored. The runtime hook
# (`build_tools/hooks/rthook_torch.py::_inject_cuda_redist_paths`) sets
# CUDA_PATH to the bundled subset at startup so triton's `find_cuda()`
# resolves to these DLLs without requiring a system-wide CUDA Toolkit install.
#
# NVCC IS NOT BUNDLED (EULA §1.1.2 #4 — developer tool, internal-use-only).
# `stage_cuda_subset.py` enforces this at staging time via a hard-reject;
# `build_release.bat` enforces it at build time via the [Bundle Prerequisites]
# probe; this spec relies on those upstream gates and does not re-check.
cuda_redist_src = project_root / 'build_tools' / 'cuda_toolkit_subset'
cuda_redist_binaries = []
cuda_redist_datas = []
if cuda_redist_src.exists():
    # Three NVRTC + CUDA Runtime DLL globs — version suffixes are
    # CUDA-Toolkit-version-specific; the staging script captures the exact
    # filenames present on the build host. Bundled as binaries so PyInstaller
    # tracks them as runtime deps.
    for _dll_pat in ('cudart64_*.dll', 'nvrtc64_*.dll',
                     'nvrtc-builtins64_*.dll'):
        for _dll in _glob.glob(str(cuda_redist_src / 'bin' / _dll_pat)):
            cuda_redist_binaries.append((_dll, 'cuda_redist/bin'))
            print(f"[SPEC] Story 18.5 — adding CUDA redistributable DLL: "
                  f"{Path(_dll).name}")

    # Device-side headers from `crt/` — required by NVRTC at compile time.
    # Small (~1 MB combined). Bundled as datas.
    _crt_dir = cuda_redist_src / 'include' / 'crt'
    if _crt_dir.exists():
        for _h in _glob.glob(str(_crt_dir / '*.h*')):
            cuda_redist_datas.append((_h, 'cuda_redist/include/crt'))
        print(f"[SPEC] Story 18.5 — bundling "
              f"{len([h for h,_ in cuda_redist_datas if 'crt' in h])} CUDA "
              f"device-side headers from {_crt_dir}")

    # EULA — load-bearing per NVIDIA EULA §1.1.2 #5 (terms-of-distribution
    # consistency). The installer.iss [Files] block additionally copies this
    # to {app}/NVIDIA_CUDA_EULA.txt at install root for end-user visibility.
    _eula = cuda_redist_src / 'EULA.txt'
    if _eula.exists():
        cuda_redist_datas.append((str(_eula), 'cuda_redist'))
        print(f"[SPEC] Story 18.5 — bundling NVIDIA CUDA Toolkit EULA")
    else:
        raise FileNotFoundError(
            f"NVIDIA EULA missing at {_eula}. The staging script "
            f"build_tools/stage_cuda_subset.py must copy %CUDA_PATH%/EULA.txt. "
            f"Bundling without the EULA is a license violation per NVIDIA EULA "
            f"§1.1.2 #5. See Story 18.5 Task 1.8 license clearance memo."
        )

    # SHA-256 hash of the EULA (written by stage_cuda_subset.py) — bundled
    # alongside so future audits can detect EULA-version drift across builds.
    _eula_hash = cuda_redist_src / 'EULA.txt.sha256'
    if _eula_hash.exists():
        cuda_redist_datas.append((str(_eula_hash), 'cuda_redist'))
else:
    print(f"[SPEC] WARNING: CUDA redistributable subset missing at "
          f"{cuda_redist_src} — run build_tools/stage_cuda_subset.py first "
          f"(see Story 18.5 Task 2; gated on Task 1.8 license clearance "
          f"sign-off). The [Bundle Prerequisites] probe in build_release.bat "
          f"should have halted the build before reaching this spec.")

# Whisper - Speech recognition
hiddenimports_whisper = collect_submodules('whisper')

# Tiktoken - Tokenizer for Whisper
hiddenimports_tiktoken = [
    'tiktoken',
    'tiktoken.core',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
]

# Audio libraries
hiddenimports_audio = [
    'pyaudio',
    'pydub',
    'soundfile',
    'sounddevice',
]

# Utilities
hiddenimports_utils = [
    'requests',
    'numpy',
    'regex',  # Required by tiktoken
    # OS trust store bridge — wired in src/myvoice/main.py immediately
    # after freeze_support() so SSL-scanning AV (Norton/Kaspersky/Avast/etc.)
    # and corporate MITM proxies don't break first-launch model resolve.
    # Top-level import in main.py would normally be discovered, but explicit
    # listing here protects against PyInstaller dropping it when main.py is
    # collected via the Analysis() entry rather than as a hidden import.
    'truststore',
]

# SciPy - Required by voice_design_studio_dialog.py
hiddenimports_scipy = collect_submodules('scipy')

# Qwen3-TTS - Core TTS engine for voice synthesis
# Note: Cannot use collect_submodules because qwen_tts imports torch which fails during build
# Must manually specify all modules
hiddenimports_qwen_tts = [
    'qwen_tts',
    'qwen_tts.cli',
    'qwen_tts.cli.demo',
    'qwen_tts.core',
    'qwen_tts.core.models',
    'qwen_tts.core.models.configuration_qwen3_tts',
    'qwen_tts.core.models.modeling_qwen3_tts',
    'qwen_tts.core.models.processing_qwen3_tts',
    'qwen_tts.core.tokenizer_12hz',
    'qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2',
    'qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2',
    'qwen_tts.core.tokenizer_25hz',
    'qwen_tts.core.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1',
    'qwen_tts.core.tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1',
    'qwen_tts.core.tokenizer_25hz.vq',
    'qwen_tts.core.tokenizer_25hz.vq.core_vq',
    'qwen_tts.core.tokenizer_25hz.vq.speech_vq',
    'qwen_tts.core.tokenizer_25hz.vq.whisper_encoder',
    'qwen_tts.inference',
    'qwen_tts.inference.qwen3_tts_model',
    'qwen_tts.inference.qwen3_tts_tokenizer',
]

# Qwen3-TTS data files (mel_filters.npz, etc.) - copy entire package
# (collect_data_files imported at top alongside collect_submodules + collect_all)
_qwen_tts_pkg = project_root / 'python310' / 'Lib' / 'site-packages' / 'qwen_tts'
qwen_tts_datas = [(str(_qwen_tts_pkg), 'qwen_tts')]

# Qwen3-TTS dependencies that need explicit inclusion
# Note: collect_submodules fails for these because they import torch which fails during build
hiddenimports_qwen_tts_deps = [
    'soundfile',
    '_soundfile',
    '_soundfile_data',
    'accelerate',
    'accelerate.utils',
    'accelerate.state',
    'accelerate.accelerator',
    'einops',
    'tqdm',
]

# Copy accelerate package and soundfile module as data (since collect_submodules fails)
_site_packages = project_root / 'python310' / 'Lib' / 'site-packages'
accelerate_datas = [(str(_site_packages / 'accelerate'), 'accelerate')]
# soundfile is a single .py file, copy it to root
soundfile_datas = [(str(_site_packages / 'soundfile.py'), '.')]
# Transformers dependency metadata - required for version checks via importlib.metadata
# Without .dist-info directories, transformers fails its dependency version checks
transformers_deps_datas = [
    (str(_site_packages / 'tqdm'), 'tqdm'),
    (str(_site_packages / 'tqdm-4.67.1.dist-info'), 'tqdm-4.67.1.dist-info'),
    (str(_site_packages / 'regex'), 'regex'),
    (str(_site_packages / 'regex-2025.9.18.dist-info'), 'regex-2025.9.18.dist-info'),
    (str(_site_packages / 'filelock-3.20.0.dist-info'), 'filelock-3.20.0.dist-info'),
    (str(_site_packages / 'huggingface_hub-0.36.0.dist-info'), 'huggingface_hub-0.36.0.dist-info'),
    (str(_site_packages / 'packaging-25.0.dist-info'), 'packaging-25.0.dist-info'),
    (str(_site_packages / 'PyYAML-6.0.3.dist-info'), 'PyYAML-6.0.3.dist-info'),
    (str(_site_packages / 'requests-2.32.5.dist-info'), 'requests-2.32.5.dist-info'),
    (str(_site_packages / 'safetensors-0.7.0.dist-info'), 'safetensors-0.7.0.dist-info'),
    (str(_site_packages / 'tokenizers-0.22.2.dist-info'), 'tokenizers-0.22.2.dist-info'),
]

# Transformers - Required by qwen_tts for model loading
# Note: Also in excludedimports to prevent build-time import crashes
hiddenimports_transformers = collect_submodules('transformers')
transformers_datas = collect_data_files('transformers')

# Jaraco modules - Required by pkg_resources at runtime
# These are vendored inside setuptools, not standalone packages
hiddenimports_jaraco = [
    'setuptools._vendor.jaraco.text',
    'setuptools._vendor.jaraco.functools',
    'setuptools._vendor.jaraco.context',
    'pkg_resources',
]

# Local TTS API (OpenAI-compatible) — tech-spec local-tts-api-v1 / F12.
# uvicorn lazily imports its protocol/loop/lifespan implementations by string,
# so PyInstaller's static analysis misses them. Without these the frozen exe
# raises ModuleNotFoundError at server start (gated by AC17). fastapi/starlette/
# pydantic are import-visible from our package, but collect_submodules keeps
# their optional submodules reachable too. Expand this list if the frozen .exe
# raises a missing-module error at API start.
hiddenimports_api_server = [
    'uvicorn',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'uvicorn.loops.auto',
    'uvicorn.loops.asyncio',
    'fastapi',
    'lameenc',
]
hiddenimports_api_server += collect_submodules('uvicorn')

# Windows-specific imports (pywin32 for Job Objects)
# Use collect_all to ensure all pywin32 modules and DLLs are included
import sys
pywin32_datas = []
pywin32_binaries = []
pywin32_hiddenimports = []

if sys.platform == 'win32':
    # Collect all pywin32 modules
    for module in ['win32job', 'win32process', 'win32api', 'win32con', 'pywintypes', 'win32com']:
        try:
            datas, binaries, hidden = collect_all(module)
            pywin32_datas.extend(datas)
            pywin32_binaries.extend(binaries)
            pywin32_hiddenimports.extend(hidden)
        except Exception:
            pass  # Module might not be importable, skip

# Combine all hidden imports
# Story 18.5 — `hiddenimports_triton` added per Block A; collect_submodules
# absorbs the triton-windows lazy-import surface (codegen + backends + jit).
hiddenimports = (
    hiddenimports_pyqt6 +
    hiddenimports_torch +
    hiddenimports_whisper +
    hiddenimports_tiktoken +
    hiddenimports_audio +
    hiddenimports_utils +
    hiddenimports_scipy +
    hiddenimports_qwen_tts +
    hiddenimports_qwen_tts_deps +
    hiddenimports_transformers +
    hiddenimports_jaraco +
    hiddenimports_triton +
    hiddenimports_api_server +
    pywin32_hiddenimports
)

# =============================================================================
# DATA FILES
# =============================================================================

# FFmpeg binaries - must be in binaries list, not datas (they're executables)
ffmpeg_binaries = []
ffmpeg_exe = project_root / 'ffmpeg' / 'ffmpeg.exe'
ffprobe_exe = project_root / 'ffmpeg' / 'ffprobe.exe'
if ffmpeg_exe.exists():
    ffmpeg_binaries.append((str(ffmpeg_exe), 'ffmpeg'))
    print(f"[SPEC] Adding ffmpeg.exe to binaries: {ffmpeg_exe}")
else:
    print(f"[SPEC] WARNING: ffmpeg.exe not found at {ffmpeg_exe}")
if ffprobe_exe.exists():
    ffmpeg_binaries.append((str(ffprobe_exe), 'ffmpeg'))
    print(f"[SPEC] Adding ffprobe.exe to binaries: {ffprobe_exe}")
else:
    print(f"[SPEC] WARNING: ffprobe.exe not found at {ffprobe_exe}")

# Application data files to bundle
datas = [
    # Icon files
    (str(icon_path / 'MyVoice.png'), 'icon'),
    (str(icon_path / 'MyVoice_Splash.png'), 'icon'),

    # Stylesheet files (if they exist)
    (str(myvoice_path / 'ui' / 'styles'), 'myvoice/ui/styles'),

    # Voice files directory (sample voices)
    (str(project_root / 'voice_files'), 'voice_files'),

    # Bundled Whisper models and assets
    (str(project_root / 'whisper_models'), 'whisper_models'),
]

# Filter out non-existent paths and log what's included
filtered_datas = []
for src, dst in datas:
    if Path(src).exists():
        filtered_datas.append((src, dst))
        print(f"[SPEC] Including: {src} -> {dst}")
    else:
        print(f"[SPEC] SKIPPING (not found): {src}")

datas = filtered_datas

# Add Whisper assets (required for transcription)
import sys
whisper_assets = None
for site_packages in sys.path:
    whisper_assets_path = Path(site_packages) / 'whisper' / 'assets'
    if whisper_assets_path.exists():
        whisper_assets = whisper_assets_path
        break

if whisper_assets:
    datas.append((str(whisper_assets), 'whisper/assets'))

# =============================================================================
# BINARY EXCLUSIONS
# =============================================================================

# Exclude large unnecessary packages to reduce size
# Note: setuptools removed from excludes - required for jaraco.text vendored module
# Note: scipy kept - required by voice_design_studio_dialog.py
excludes = [
    'matplotlib',
    'pandas',
    'jupyter',
    'notebook',
    'IPython',
    'pytest',
    'sphinx',
    'wheel',
    'pip',
]

# =============================================================================
# ANALYSIS
# =============================================================================

# Modules to exclude from import analysis (prevents crash during build)
# These will still be included via hiddenimports and datas
excludedimports = [
    'torch',
    'torch._C',
    'transformers',
    'qwen_tts',
]

a = Analysis(
    [str(myvoice_path / 'main.py')],
    pathex=[str(src_path)],
    # Story 18.5 — `cuda_redist_binaries` adds the CUDA Toolkit redistributable
    # subset (NVIDIA EULA Attachment A scope only); see Block C above.
    binaries=(
        pywin32_binaries + ffmpeg_binaries + torch_binaries + cuda_redist_binaries
    ),
    # Story 18.5 — `triton_datas` + `python_headers_datas` + `python_libs_datas`
    # + `cuda_redist_datas` add the four bundling components per Blocks A/B/C.
    datas=(
        datas + pywin32_datas + torch_datas + torch_serialized_patterns_datas
        + qwen_tts_datas + accelerate_datas + soundfile_datas
        + transformers_deps_datas + transformers_datas
        + triton_datas + python_headers_datas + python_libs_datas
        + cuda_redist_datas
    ),
    hiddenimports=hiddenimports,
    hookspath=[],
    # Story 18.5 — `'triton': 'pyz+py'` makes triton-windows's Python sources
    # available BOTH in the PYZ archive AND on the filesystem at `_internal/
    # triton/` — triton's lazy-import surface expects the on-disk tree to
    # resolve `importlib.import_module()` calls at runtime. Mirrors the
    # existing torch / transformers / qwen_tts entries.
    module_collection_mode={
        'torch': 'pyz+py',
        'transformers': 'pyz+py',
        'qwen_tts': 'pyz+py',
        'triton': 'pyz+py',
    },
    hooksconfig={},
    runtime_hooks=[str(spec_dir / 'hooks' / 'rthook_torch.py')],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# =============================================================================
# PYZ (Python Archive)
# =============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# =============================================================================
# SPLASH SCREEN
# =============================================================================

# Note: PyInstaller's splash screen requires tkinter which is not included in
# portable Python distributions. Instead, we'll implement a QSplashScreen in
# the application code itself using PyQt6 for better control and compatibility.
# splash = Splash(
#     str(icon_path / 'MyVoice_Splash.png'),
#     binaries=a.binaries,
#     datas=a.datas,
#     text_pos=(10, 260),
#     text_size=10,
#     text_color='white',
#     text_default='Loading MyVoice...',
#     minify_script=True,
#     always_on_top=True,
# )

# =============================================================================
# EXECUTABLE
# =============================================================================

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,  # One-folder mode for better performance
    name='MyVoice',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX disabled to reduce antivirus false positives
    console=False,  # Windowed application (no console)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path / 'MyVoice.ico'),  # Use .ico file directly
)

# =============================================================================
# COLLECT (One-Folder Distribution)
# =============================================================================

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # UPX disabled to reduce antivirus false positives
    upx_exclude=[],
    name='MyVoice',
)

# =============================================================================
# BUILD NOTES - PORTABLE DISTRIBUTION
# =============================================================================

"""
Portable Distribution Structure:
    dist/MyVoice/
    ├── MyVoice.exe              # Main executable
    ├── _internal/               # Python runtime and dependencies (bundled by PyInstaller)
    │   ├── python310.dll
    │   ├── python3.dll
    │   ├── torch/
    │   ├── whisper/
    │   ├── ffmpeg/             # FFmpeg binaries
    │   └── ...
    ├── config/                  # User settings (created on first run)
    │   └── settings.json       # Application configuration
    ├── logs/                    # Application logs (created on first run)
    │   └── myvoice.log
    ├── voice_files/            # Voice samples (copied by build script)
    │   └── ...
    ├── whisper_models/         # Whisper AI models (created on first run)
    └── README.txt              # User documentation

Portable Features:
    - ALL user data stored in application directory (not %APPDATA%)
    - No installation required
    - Can be moved to any location
    - Can run from USB drive
    - Multiple instances supported (different folders)
    - Clean removal by deleting folder

Python Bundling:
    - Python 3.10 runtime automatically bundled
    - Users do NOT need Python installed
    - All dependencies in _internal/
    - Fully self-contained executable

Path Resolution:
    - Uses portable_paths.py utility for all file paths
    - Resolves paths relative to MyVoice.exe location
    - Works in both frozen and development modes
    - sys.executable.parent used as base directory

Size Estimates:
    - Base application: ~250MB
    - With default voices: ~280MB
    - With Whisper models: ~420MB
    - With Qwen3-TTS models (downloaded on first use):
        - Quality tier (1.7B): ~3.4GB
        - Small tier (0.6B): ~1.2GB

Distribution Strategy:
    1. Build with: python build_tools/build_portable.py
    2. Compress dist/MyVoice/ to MyVoice-Portable-v2.2.0.zip
    3. Users extract and run MyVoice.exe
"""
