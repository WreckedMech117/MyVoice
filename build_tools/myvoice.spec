# -*- mode: python ; coding: utf-8 -*-
"""
MyVoice PyInstaller Specification File
Builds standalone Windows executable with splash screen and icon

Build Command:
    pyinstaller myvoice.spec

Output:
    dist/MyVoice/MyVoice.exe
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules
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

# Torch DLL binaries - collected manually to avoid import issues
import glob as _glob
torch_binaries = []
_torch_lib = project_root / 'python310' / 'Lib' / 'site-packages' / 'torch' / 'lib'
if _torch_lib.exists():
    for _dll in _glob.glob(str(_torch_lib / '*.dll')):
        torch_binaries.append((_dll, 'torch/lib'))
        print(f"[SPEC] Adding torch DLL: {Path(_dll).name}")
torch_datas = []

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
from PyInstaller.utils.hooks import collect_data_files
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
    binaries=pywin32_binaries + ffmpeg_binaries + torch_binaries,
    datas=datas + pywin32_datas + torch_datas + qwen_tts_datas + accelerate_datas + soundfile_datas + transformers_deps_datas + transformers_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    module_collection_mode={'torch': 'pyz+py', 'transformers': 'pyz+py', 'qwen_tts': 'pyz+py'},
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
    2. Compress dist/MyVoice/ to MyVoice-Portable-v2.1.0.zip
    3. Users extract and run MyVoice.exe
"""
