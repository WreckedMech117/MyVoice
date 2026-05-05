# MyVoice Executable Size Optimization Guide

This guide documents strategies for reducing MyVoice executable size from **1-2GB** down to **250-400MB**.

---

## 📊 Size Analysis Breakdown

### Before Optimization (Default Build)
| Component | Size | Notes |
|-----------|------|-------|
| PyTorch (CUDA) | ~2.5 GB | Includes GPU support |
| Whisper Models | ~1.5 GB | If bundled (medium model) |
| PyQt6 | ~100 MB | GUI framework |
| NumPy | ~50 MB | Math library |
| Other dependencies | ~50 MB | requests, pyaudio, etc. |
| MyVoice code | ~5 MB | Application logic |
| **Total (worst case)** | **~4.2 GB** | With CUDA + bundled models |

### After Optimization (Production Build)
| Component | Size | Strategy Applied |
|-----------|------|------------------|
| PyTorch (CPU-only) | ~150 MB | ✅ Removed CUDA (~2.3GB saved) |
| Whisper Models | 0 MB | ✅ Download at runtime |
| PyQt6 | ~100 MB | Auto-optimized by PyInstaller |
| NumPy | ~50 MB | Required, minimal |
| Other dependencies | ~50 MB | Minimal required set |
| MyVoice code | ~5 MB | Application logic |
| **Total (uncompressed)** | **~355 MB** | 91% reduction |
| **Total (with UPX)** | **~250 MB** | Additional 30% compression |

**Size Reduction: 95% (4.2GB → 250MB)**

---

## 🎯 Optimization Strategies

### 1. Use CPU-Only PyTorch (Highest Impact)

**Problem:** Default PyTorch includes CUDA libraries (~2.3GB) even if GPU isn't used.

**Solution:** Install CPU-only version

```bash
# Uninstall existing torch
pip uninstall torch torchvision torchaudio

# Install CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**In requirements-production.txt:**
```txt
torch>=2.0.0
--extra-index-url https://download.pytorch.org/whl/cpu
```

**Size Savings:** ~2.3GB

**Trade-offs:**
- ✅ No GPU support (not needed for MyVoice)
- ✅ Faster CPU inference on modern processors
- ✅ Better compatibility across systems

---

### 2. Download Whisper Models at Runtime (High Impact)

**Problem:** Bundling Whisper models adds 75MB-3GB depending on model size.

**Solution:** Download models on first run (Whisper does this automatically)

**Implementation:**
- Remove models from bundle
- Whisper downloads to `~/.cache/whisper/` on first use
- Show progress dialog during first-time download
- Cache models for subsequent runs

**Model Size Reference:**
| Model | Parameters | Size | Quality | Speed |
|-------|-----------|------|---------|-------|
| tiny | 39M | 75 MB | Lowest | Fastest |
| base | 74M | 140 MB | **Recommended** | Fast |
| small | 244M | 470 MB | Good | Medium |
| medium | 769M | 1.5 GB | Better | Slow |
| large | 1550M | 3.0 GB | Best | Slowest |

**Recommended Default:** `base` model (good balance)

**Size Savings:** 75MB - 3GB (depending on model)

**Implementation Notes:**
```python
# MyVoice should detect first run and show dialog
if not whisper_model_cached():
    show_download_progress_dialog()
    whisper.load_model("base")  # Auto-downloads
```

---

### 3. Enable UPX Compression (Medium Impact)

**Problem:** Executable contains uncompressed DLLs and binaries.

**Solution:** Use UPX (Ultimate Packer for eXecutables)

**Installation:**
```bash
# Download from https://upx.github.io/
# Extract to C:\Program Files\upx\
# Add to PATH
```

**Already Enabled in myvoice.spec:**
```python
exe = EXE(
    # ...
    upx=True,  # ✅ Already enabled
    # ...
)
```

**Size Savings:** ~30-50% of uncompressed size

**Trade-offs:**
- ✅ Transparent decompression
- ⚠️ Slightly slower startup (< 1 second)
- ⚠️ May trigger antivirus false positives

**Antivirus Note:** If UPX causes issues, set `upx=False` in spec file.

---

### 4. Exclude Unused Python Modules (Low-Medium Impact)

**Problem:** PyInstaller may include unused standard library modules.

**Solution:** Explicitly exclude in spec file

**Already Configured in myvoice.spec:**
```python
excludes=[
    'matplotlib',  # Plotting library (~50MB)
    'scipy',       # Scientific computing (~40MB)
    'pandas',      # Data analysis (~30MB)
    'PIL',         # Image processing (~10MB)
    'cv2',         # Computer vision (~50MB)
    'tkinter',     # Alternative GUI framework (~5MB)
    '_tkinter',
    'unittest',    # Testing framework (~5MB)
    'test',
    'pytest',
]
```

**Size Savings:** ~190MB total

**Add More Exclusions:** If you identify unused modules during testing.

---

### 5. Strip Debug Symbols (Low Impact)

**Problem:** Binaries contain debugging information.

**Solution:** Enable stripping in spec file

**Current Setting:**
```python
exe = EXE(
    # ...
    strip=False,  # Set to True for production
    # ...
)
```

**For Production:** Change to `strip=True`

**Size Savings:** ~10-20MB

**Trade-offs:**
- ✅ Smaller size
- ❌ Harder to debug crash reports
- **Recommendation:** Use `strip=False` during development, `strip=True` for release

---

### 6. Optimize PyQt6 Imports (Low Impact)

**Problem:** Importing all of PyQt6 includes unused modules.

**Solution:** Import only what's needed (already done in code)

**Good Practice:**
```python
# ✅ Import only used modules
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon

# ❌ Avoid broad imports
from PyQt6.QtWidgets import *
```

**PyInstaller Optimization:** Automatically detects used modules

**Size Savings:** PyInstaller handles this automatically, ~20-30MB

---

### 7. Remove Development Dependencies (Low Impact)

**Problem:** Build environment may include dev tools.

**Solution:** Use clean environment with production requirements

**Process:**
```bash
# Create clean environment
python -m venv venv-production
venv-production\Scripts\activate

# Install ONLY production dependencies
pip install -r build_tools/requirements-production.txt

# Install build tools
pip install pyinstaller>=6.0.0

# Build
cd build_tools
pyinstaller myvoice.spec
```

**Size Savings:** ~50-100MB (prevents accidental inclusion)

---

## 📋 Optimization Checklist

Use this checklist when building for production:

### Pre-Build Optimization
- [ ] Clean virtual environment created
- [ ] Installed CPU-only PyTorch (not CUDA version)
- [ ] Installed requirements-production.txt (not requirements.txt)
- [ ] Verified no development packages installed
- [ ] UPX installed and in PATH (optional)

### Build Configuration
- [ ] Reviewed `myvoice.spec` excludes list
- [ ] Confirmed `upx=True` in spec file
- [ ] Set `strip=True` for release builds
- [ ] Removed model bundling from datas
- [ ] Verified hiddenimports are minimal

### Post-Build Verification
- [ ] Measured final executable size
- [ ] Tested application launches
- [ ] Verified Whisper downloads models on first run
- [ ] Tested on clean Windows system
- [ ] Checked for antivirus false positives

---

## 🎯 Target Sizes by Configuration

### Configuration Matrix

| Config | PyTorch | Models | UPX | Expected Size |
|--------|---------|--------|-----|---------------|
| **Development** (CUDA + bundled) | CUDA | Bundled | No | ~4GB |
| **Standard** (CPU + bundled) | CPU | Bundled | No | ~1.5GB |
| **Recommended** (CPU + runtime) | CPU | Runtime | No | ~350MB |
| **Optimized** (CPU + runtime) | CPU | Runtime | Yes | **~250MB** |

---

## 🔧 Advanced Optimization (Optional)

### Use PyInstaller Bootloader Customization

For advanced users, custom bootloader can reduce overhead:

```bash
# Rebuild bootloader with minimal features
git clone https://github.com/pyinstaller/pyinstaller.git
cd pyinstaller/bootloader
python ./waf all
```

**Size Savings:** ~5-10MB

**Complexity:** High, only for advanced users

---

### Alternative: faster-whisper

Consider using `faster-whisper` instead of `openai-whisper`:

```bash
pip install faster-whisper
```

**Advantages:**
- 4x faster inference
- Lower memory usage
- Smaller model files (quantized)

**Trade-offs:**
- Requires code changes
- Different API
- May affect accuracy slightly

**Size Savings:** ~30-40% smaller models

---

## 📈 Measuring Executable Size

### Before Build
```bash
# Check installed package sizes
pip list --format=freeze | xargs pip show | grep -E 'Location:|Name:|Version:'
```

### After Build
```bash
# Windows
dir /s build_tools\dist\MyVoice

# PowerShell
Get-ChildItem -Path build_tools\dist\MyVoice -Recurse | Measure-Object -Property Length -Sum
```

### Expected Output
```
Directory: build_tools\dist\MyVoice

Total Size: ~250-350 MB (optimized)
File Count: ~50-100 files
Folders: MyVoice.exe + _internal\
```

---

## 🐛 Troubleshooting Size Issues

### Executable Still Too Large?

1. **Check for CUDA:**
   ```bash
   # Search for CUDA DLLs in dist folder
   dir /s build_tools\dist\MyVoice\*cuda*.dll
   ```
   If found, reinstall CPU-only torch.

2. **Check for Bundled Models:**
   ```bash
   # Search for Whisper models
   dir /s build_tools\dist\MyVoice\*.pt
   ```
   Models should NOT be in dist folder.

3. **Identify Large Files:**
   ```powershell
   Get-ChildItem -Path build_tools\dist\MyVoice -Recurse |
   Sort-Object Length -Descending |
   Select-Object -First 20 FullName, @{Name="SizeMB";Expression={$_.Length/1MB}}
   ```

4. **Analyze with PyInstaller Archive Viewer:**
   ```bash
   pyi-archive_viewer build_tools\dist\MyVoice\MyVoice.exe
   ```

---

## ✅ Success Criteria

Your optimized build is successful when:

- ✅ Total size < 400MB (preferably ~250-300MB)
- ✅ Application launches without errors
- ✅ All features work correctly
- ✅ Whisper downloads models on first run
- ✅ No CUDA libraries in dist folder
- ✅ Startup time < 5 seconds
- ✅ Works on clean Windows system

---

## 📚 Additional Resources

- **PyTorch CPU Builds:** https://pytorch.org/get-started/locally/
- **PyInstaller Optimization:** https://pyinstaller.org/en/stable/when-things-go-wrong.html
- **UPX Documentation:** https://upx.github.io/
- **Whisper Models:** https://github.com/openai/whisper#available-models-and-languages

---

## 🎉 Expected Results

Following this guide should achieve:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Size** | 4.2 GB | 250 MB | **95% reduction** |
| **Startup** | 15-30s | 3-5s | **80% faster** |
| **Distribution** | Impractical | Easy | Download-friendly |
| **User Experience** | Poor | Excellent | Professional |

---

*Last Updated: 2026-02-08*
*For MyVoice v2.0.0 Release*

## V2 Considerations

### Qwen3-TTS Model Management

V2 uses three Qwen3-TTS 1.7B models with lazy loading:

| Model | Purpose | Size | When Loaded |
|-------|---------|------|-------------|
| Base | Voice Cloning | ~3.4GB | When cloned voice selected |
| CustomVoice | Bundled Voices | ~3.4GB | When bundled voice selected |
| VoiceDesign | Voice Design | ~3.4GB | When designed voice selected |

**Key Optimization:** Only ONE model loads at a time. Switching voice types unloads the current model before loading the new one.

**NOT Bundled:** Models download on first use from Hugging Face. This keeps installer small (~300MB) but requires internet for first use of each voice type.

### Memory Considerations

- **Runtime memory:** ~4GB when model loaded (3.4GB model + overhead)
- **NFR11 target:** <4GB RAM during active use
- **Recommendation:** 16GB system RAM for smooth operation
