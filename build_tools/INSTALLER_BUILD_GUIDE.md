# MyVoice Installer Build Guide

Complete guide for building the MyVoice Windows installer using Inno Setup.

---

## Prerequisites

### 1. Install Inno Setup 6.x

**Download:** a

**Installation:**
```
1. Download Inno Setup 6.x (current: 6.3.3)
2. Run installer
3. Accept license agreement
4. Install to default location: C:\Program Files (x86)\Inno Setup 6
5. Complete installation
```

**Verify Installation:**
```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /?
```

Should display Inno Setup compiler version and options.

---

### 2. Build MyVoice Executable First

Before creating the installer, you **must** build the executable:

```bash
# Navigate to build tools
cd G:\MyVoicePublic\build_tools

# Build with PyInstaller
pyinstaller myvoice.spec

# Verify output exists
dir dist\MyVoice\MyVoice.exe
```

**Expected Output:**
```
G:\MyVoicePublic\build_tools\dist\MyVoice\
├── MyVoice.exe
└── _internal\
    └── [hundreds of bundled files]
```

---

## Building the Installer

### Method 1: GUI (Recommended for First Build)

1. **Open Inno Setup Compiler**
   - Start Menu > Inno Setup 6 > Inno Setup Compiler

2. **Open installer.iss**
   - File > Open
   - Navigate to: `G:\MyVoicePublic\build_tools\installer.iss`
   - Click Open

3. **Review Configuration**
   - Check paths in [Files] section
   - Verify icon paths exist
   - Review version numbers

4. **Compile**
   - Build > Compile (or press Ctrl+F9)
   - Watch output window for progress
   - Check for errors or warnings

5. **Output Location**
   - Success: `G:\MyVoicePublic\installer_output\MyVoice-Setup-v2.0.0.exe`
   - Size: ~250-400MB (same as dist folder)

---

### Method 2: Command Line (Automation)

```bash
# Navigate to build tools
cd G:\MyVoicePublic\build_tools

# Compile installer
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

# Check output
dir ..\installer_output\MyVoice-Setup-v2.0.0.exe
```

**Batch Script** (`build_tools\build_installer.bat`):
```batch
@echo off
echo Building MyVoice Installer...
echo.

REM Check if Inno Setup is installed
if not exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    echo ERROR: Inno Setup 6 not found!
    echo Please install from: https://jrsoftware.org/isdl.php
    pause
    exit /b 1
)

REM Check if MyVoice.exe exists
if not exist "dist\MyVoice\MyVoice.exe" (
    echo ERROR: MyVoice.exe not found in dist\MyVoice\
    echo Please build executable first: pyinstaller myvoice.spec
    pause
    exit /b 1
)

REM Compile installer
echo Compiling installer with Inno Setup...
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! Installer created:
    echo %~dp0..\installer_output\MyVoice-Setup-v2.0.0.exe
    echo.
    dir "%~dp0..\installer_output\MyVoice-Setup-v2.0.0.exe"
) else (
    echo.
    echo ERROR: Installer compilation failed!
    echo Check the output above for errors.
)

pause
```

Save as `G:\MyVoicePublic\build_tools\build_installer.bat`

**Usage:**
```bash
cd G:\MyVoicePublic\build_tools
build_installer.bat
```

---

## Installer Configuration

### Key Settings in installer.iss

#### Application Identity
```pascal
#define MyAppName "MyVoice"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "MyVoice Development Team"
```

**To Update Version:**
1. Open `installer.iss`
2. Change `#define MyAppVersion "2.0.0"` to new version
3. Recompile

---

#### Installation Paths
```pascal
DefaultDirName={autopf}\{#MyAppName}
```

**Default:** `C:\Program Files\MyVoice`

Users can change during installation.

---

#### Compression
```pascal
Compression=lzma2/ultra64
SolidCompression=yes
```

**Settings:**
- `lzma2/ultra64` - Maximum compression (slower build, smallest size)
- `lzma2/max` - Good compression (faster build)
- `lzma2/normal` - Balanced (recommended for testing)

**Trade-offs:**
- Ultra64: ~10-15% smaller, 2-3x slower build
- Normal: Faster build, slightly larger installer

---

#### Visual Branding
```pascal
WizardImageFile=..\src\icon\MyVoice_Splash.png
WizardSmallImageFile=..\src\icon\MyVoice.png
SetupIconFile=..\src\icon\MyVoice.png
```

**Image Requirements:**
- **WizardImageFile:** 164x314 pixels (large left-side image)
- **WizardSmallImageFile:** 55x55 pixels (top-right corner)
- **SetupIconFile:** Any size, converted to .ico format

**Current Setup:**
- Uses `MyVoice_Splash.png` for wizard
- Uses `MyVoice.png` for icon

---

## Testing the Installer

### Pre-Installation Checks

Before distributing, test the installer:

#### 1. Verify File Integrity

```bash
# Check installer size (should be 250-400MB)
dir G:\MyVoicePublic\installer_output\MyVoice-Setup-v2.0.0.exe

# Run virus scan (optional but recommended)
# Windows Defender:
"C:\Program Files\Windows Defender\MpCmdRun.exe" -Scan -ScanType 3 -File "G:\MyVoicePublic\installer_output\MyVoice-Setup-v2.0.0.exe"
```

---

#### 2. Test Installation

**On Development Machine:**
```
1. Run MyVoice-Setup-v2.0.0.exe
2. Follow installation wizard
3. Install to test directory (e.g., C:\Temp\MyVoice-Test)
4. Verify installation completes
5. Test launching application
6. Test uninstallation
```

---

#### 3. Test on Clean Windows System

**Best Practice:** Use Windows Sandbox or VM

**Windows Sandbox (Windows 10 Pro+):**
```
1. Enable Windows Sandbox:
   - Settings > Apps > Optional Features
   - Add "Windows Sandbox"

2. Start Windows Sandbox

3. Copy installer to Sandbox:
   - Drag & drop MyVoice-Setup-v2.0.0.exe into Sandbox

4. Run installer in Sandbox

5. Test application functionality

6. Close Sandbox (everything cleaned up automatically)
```

**VM Testing (VMware/VirtualBox):**
```
1. Create fresh Windows 10/11 VM
2. Take snapshot before testing
3. Install MyVoice
4. Test all features
5. Test uninstallation
6. Revert to snapshot for clean testing
```

---

### Installation Test Checklist

- [ ] Installer runs without admin prompt
- [ ] License agreement displays correctly
- [ ] Readme file displays before installation
- [ ] Installation directory can be customized
- [ ] Desktop shortcut option works
- [ ] Start Menu shortcut option works
- [ ] Installation completes without errors
- [ ] Application launches after installation
- [ ] Application icon displays correctly
- [ ] All features work (TTS, transcription, etc.)
- [ ] Application appears in Add/Remove Programs
- [ ] Uninstaller runs successfully
- [ ] All files removed after uninstall
- [ ] Shortcuts removed after uninstall
- [ ] No registry entries left behind

---

## Silent Installation

The installer supports silent (unattended) installation:

### Silent Install
```bash
# No UI, but shows progress bar
MyVoice-Setup-v2.0.0.exe /SILENT

# Install to specific directory
MyVoice-Setup-v2.0.0.exe /SILENT /DIR="C:\MyCustomPath"
```

### Very Silent Install
```bash
# Completely hidden, no UI at all
MyVoice-Setup-v2.0.0.exe /VERYSILENT

# With custom directory
MyVoice-Setup-v2.0.0.exe /VERYSILENT /DIR="C:\MyCustomPath"

# Suppress restart prompts
MyVoice-Setup-v2.0.0.exe /VERYSILENT /NORESTART
```

### Additional Options
```bash
# Create desktop icon (if optional in installer)
MyVoice-Setup-v2.0.0.exe /SILENT /TASKS="desktopicon"

# Don't create desktop icon
MyVoice-Setup-v2.0.0.exe /SILENT /TASKS="!desktopicon"

# Log installation to file
MyVoice-Setup-v2.0.0.exe /SILENT /LOG="C:\install-log.txt"
```

---

## Troubleshooting

### Common Build Errors

#### Error: "Source file not found"

```
Error: Source file "...\dist\MyVoice\MyVoice.exe" not found
```

**Solution:**
1. Build executable first: `pyinstaller myvoice.spec`
2. Verify path in installer.iss matches actual dist location
3. Check for typos in [Files] section

---

#### Error: "Icon file not found"

```
Error: SetupIconFile: Unable to open file "...\MyVoice.png"
```

**Solution:**
1. Verify icon files exist:
   - `G:\MyVoicePublic\src\icon\MyVoice.png`
   - `G:\MyVoicePublic\src\icon\MyVoice_Splash.png`
2. Check paths in [Setup] section
3. Use absolute paths if relative paths fail

---

#### Warning: "Image size not recommended"

```
Warning: WizardImageFile size is not 164x314
```

**Not Critical:** Inno Setup will resize automatically

**Fix (Optional):** Resize images to recommended dimensions

---

#### Error: "Disk full"

```
Error: Not enough disk space to create installer
```

**Solution:**
1. Free up disk space (need ~500MB for temp files)
2. Change temp directory: Set TEMP environment variable
3. Build on different drive with more space

---

### Installer Runtime Issues

#### Installation fails with "Access Denied"

**Cause:** Installing to protected location without admin rights

**Solution:**
- Run installer as Administrator
- Or install to user directory (not Program Files)

---

#### Application won't launch after install

**Possible Causes:**
1. Missing Visual C++ Redistributable
2. Windows Defender blocked files
3. Corrupted installation

**Solutions:**
```
1. Install VC++ Redistributable:
   - Download from Microsoft
   - Or include in installer (advanced)

2. Check Windows Defender:
   - Windows Security > Virus & threat protection
   - Check quarantine for MyVoice files
   - Add exclusion if needed

3. Reinstall:
   - Uninstall completely
   - Delete installation folder
   - Reinstall
```

---

#### Uninstaller doesn't remove all files

**Normal Behavior:** Uninstaller keeps:
- User settings (by design)
- Downloaded Whisper models (by design)
- Log files created during runtime

**To Remove Completely:**
```
1. Run uninstaller
2. Manually delete:
   - %LOCALAPPDATA%\MyVoice
   - %APPDATA%\MyVoice
   - %USERPROFILE%\.cache\whisper (if no other apps use it)
```

---

## Advanced Customization

### Add Custom Wizard Pages

Edit `[Code]` section in installer.iss:

```pascal
procedure InitializeWizard;
var
  CustomPage: TWizardPage;
begin
  CustomPage := CreateCustomPage(wpLicense,
    'Configuration', 'Configure MyVoice settings');

  // Add custom controls here
end;
```

See Inno Setup documentation for examples.

---

### Check for Prerequisites

Example: Check for Visual C++ Redistributable

```pascal
function VCRedistNeedsInstall: Boolean;
var
  Version: String;
begin
  Result := not RegQueryStringValue(HKLM,
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
    'Version', Version);
end;
```

Already included in installer.iss but not actively used.

---

### Multilingual Support

Add translations in `[Languages]` section:

```pascal
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
```

Inno Setup includes 40+ language files.

---

## Distribution

### Recommended File Naming

```
MyVoice-Setup-v2.0.0.exe          # Standard release (models download on first use)
```

**Note:** V2 uses lazy model loading. Qwen3-TTS models (~3.4GB each) download automatically
on first use. This keeps the installer small (~300MB) while supporting all features.

---

### Checksums

Generate checksums for distribution:

```powershell
# SHA256
Get-FileHash MyVoice-Setup-v2.0.0.exe -Algorithm SHA256 | Format-List

# MD5 (for compatibility)
Get-FileHash MyVoice-Setup-v2.0.0.exe -Algorithm MD5 | Format-List
```

Include in release notes for users to verify integrity.

---

### Code Signing (Optional - See EXE3.2)

Sign the installer for professional distribution:

```bash
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com MyVoice-Setup-v2.0.0.exe
```

Prevents "Unknown Publisher" warnings.

---

## Automated Build Script

Complete automation script combining PyInstaller + Inno Setup:

**`build_tools\build_release.bat`:**

```batch
@echo off
setlocal enabledelayedexpansion

echo ========================================
echo MyVoice Release Builder
echo ========================================
echo.

REM Step 1: Clean previous builds
echo [1/4] Cleaning previous builds...
if exist "dist" rd /s /q "dist"
if exist "build" rd /s /q "build"
if exist "..\installer_output" rd /s /q "..\installer_output"
echo Done.
echo.

REM Step 2: Build executable with PyInstaller
echo [2/4] Building executable with PyInstaller...
pyinstaller myvoice.spec
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyInstaller build failed!
    pause
    exit /b 1
)
echo Done.
echo.

REM Step 3: Build installer with Inno Setup
echo [3/4] Building installer with Inno Setup...
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Inno Setup build failed!
    pause
    exit /b 1
)
echo Done.
echo.

REM Step 4: Generate checksums
echo [4/4] Generating checksums...
cd ..\installer_output
for %%F in (*.exe) do (
    echo Calculating SHA256 for %%F...
    certutil -hashfile "%%F" SHA256 > "%%F.sha256"
)
echo Done.
echo.

echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Output:
dir /b *.exe
echo.
echo Location: %CD%
echo.

pause
```

**Usage:**
```bash
cd G:\MyVoicePublic\build_tools
build_release.bat
```

---

## Summary

✅ **Installer Features:**
- Professional installation wizard
- Custom branding with splash screen
- License agreement and readme
- Start Menu and Desktop shortcuts
- Clean uninstallation
- Silent install support
- Progress indicators

✅ **Build Process:**
1. Build executable with PyInstaller
2. Compile installer with Inno Setup
3. Test on clean system
4. Generate checksums
5. (Optional) Code sign
6. Distribute

✅ **Next Steps:**
- Test installer thoroughly
- Create documentation (EXE4.1)
- Consider code signing (EXE3.2)
- Set up automated builds (EXE3.1)

---

*For questions about Inno Setup, see: https://jrsoftware.org/ishelp/*
