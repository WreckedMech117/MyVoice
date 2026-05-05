# MyVoice V2 Release Checklist

Complete this checklist before distributing a new MyVoice release.

**Current Version: 2.0.0**

---

## Pre-Build Phase

### Code Quality
- [ ] All features tested and working
- [ ] No critical bugs or issues
- [ ] Code reviewed and approved
- [ ] Documentation updated
- [ ] Changelog updated

### Version Management
- [ ] Version number decided (major.minor.patch)
- [ ] Version updated: `python version.py bump <component>`
- [ ] All files have correct version number
- [ ] Git repository tagged with version

### Dependencies
- [ ] All dependencies up-to-date
- [ ] Security vulnerabilities checked
- [ ] License compliance verified
- [ ] Third-party notices updated

---

## Build Phase

### Environment Setup
- [ ] Clean build environment: `clean_build.bat`
- [ ] Virtual environment activated (if using)
- [ ] Production requirements installed: `requirements-production.txt`
- [ ] PyInstaller installed: `pip install pyinstaller>=6.0.0`
- [ ] Inno Setup 6.x installed

### Build Execution
- [ ] Run: `build_release.bat`
- [ ] Build completed without errors
- [ ] Build logs reviewed
- [ ] No warnings about missing modules

---

## Post-Build Validation

### Automated Tests
- [ ] Run: `test_release.bat`
- [ ] All automated tests passed
- [ ] Executable size reasonable (~250-400MB)
- [ ] Installer size reasonable (~250-400MB)
- [ ] Checksums generated (SHA256, MD5)

### Manual Testing - Executable
- [ ] Executable launches without errors
- [ ] Splash screen displays correctly
- [ ] Main window appears with proper icon
- [ ] All UI elements render correctly
- [ ] No console window appears (windowed mode)
- [ ] Application closes cleanly

### Manual Testing - Core Features
- [ ] Audio device detection works
- [ ] Monitor speaker output functional
- [ ] Virtual microphone output functional
- [ ] Dual audio routing (monitor + virtual mic simultaneously)
- [ ] Settings save/load correctly
- [ ] Settings persist after restart

### Manual Testing - V2 Voice Features
- [ ] Bundled voices appear in voice selector (9 timbres)
- [ ] Bundled voice TTS generation works
- [ ] Voice Design dialog opens
- [ ] Voice Design preview generates audio
- [ ] Voice Design save creates new voice profile
- [ ] Voice Clone dialog opens
- [ ] Voice Clone embedding extraction works
- [ ] Voice Clone save creates new voice profile
- [ ] Voice library shows all voice types with correct icons
- [ ] Deleting user-created voices works

### Manual Testing - V2 Emotion Control
- [ ] Emotion button row visible (5 presets)
- [ ] Clicking emotion changes selection
- [ ] F1-F5 keyboard shortcuts work
- [ ] Emotion affects TTS output (audible difference)
- [ ] Emotion buttons disabled for cloned voices
- [ ] Custom emotion (Settings → TTS) works

### Manual Testing - V2 Quick Speak
- [ ] Quick Speak button visible in main window
- [ ] Quick Speak menu shows configured phrases
- [ ] Selecting phrase triggers generation
- [ ] Quick Speak uses current voice and emotion

### Manual Testing - V2 TTS/Models
- [ ] Model loading indicator shows during first load
- [ ] Model loads within 30 seconds
- [ ] Switching voice types triggers model change
- [ ] Lazy loading works (only one model in memory)
- [ ] Streaming TTS works (audio starts within 2 seconds)
- [ ] Batch fallback works if streaming fails

### Manual Testing - Transcription
- [ ] Speech recognition works (Whisper)
- [ ] Background transcription functional
- [ ] Whisper model downloads on first use

### Manual Testing - Installer
- [ ] Installer launches without errors
- [ ] Splash screen displays during installation
- [ ] License agreement displays
- [ ] Readme displays correctly
- [ ] Installation directory can be customized
- [ ] Desktop shortcut option works
- [ ] Start Menu shortcuts created
- [ ] Application launches after installation
- [ ] Application appears in Add/Remove Programs
- [ ] Uninstaller works correctly
- [ ] All files removed after uninstall
- [ ] No registry entries left behind

---

## Clean System Testing

### Windows 10 Test
- [ ] Tested on Windows 10 (Home or Pro)
- [ ] Fresh install (no Python, no dev tools)
- [ ] Installer runs without admin issues
- [ ] Application works fully
- [ ] Uninstall works cleanly

### Windows 11 Test
- [ ] Tested on Windows 11
- [ ] Fresh install (no Python, no dev tools)
- [ ] Application works fully
- [ ] No compatibility warnings

### Different Configurations
- [ ] Tested with no internet (after first model download)
- [ ] Tested on different screen resolutions
- [ ] Tested with different audio devices

---

## Distribution Preparation

### File Organization
- [ ] Release folder created: `MyVoice-v{version}`
- [ ] Installer copied to release folder
- [ ] Checksums copied to release folder
- [ ] README.txt included
- [ ] LICENSE.txt included
- [ ] Release notes created

### Checksums & Signatures
- [ ] SHA256 checksum generated
- [ ] MD5 checksum generated (for compatibility)
- [ ] Checksums verified manually
- [ ] (Optional) Installer code signed

### Documentation
- [ ] User documentation complete
- [ ] Installation guide up-to-date
- [ ] Troubleshooting guide reviewed
- [ ] FAQ updated
- [ ] System requirements documented
- [ ] Known issues documented

---

## Security & Compliance

### Security Checks
- [ ] Virus scan performed (Windows Defender)
- [ ] No false positives from antivirus
- [ ] No hardcoded credentials or secrets
- [ ] No debug code or print statements
- [ ] Error messages don't leak sensitive info

### License Compliance
- [ ] All third-party licenses included
- [ ] License notices in installer
- [ ] Copyright notices up-to-date
- [ ] Attribution for dependencies

---

## Release Distribution

### Upload Preparation
- [ ] Final installer tested one more time
- [ ] File naming convention: `MyVoice-Setup-v{version}.exe`
- [ ] Release notes finalized
- [ ] Screenshots/demo video prepared (if applicable)

### Distribution Checklist
- [ ] Upload to release platform (GitHub, website, etc.)
- [ ] Checksums posted publicly
- [ ] Release notes published
- [ ] Download links tested
- [ ] Version announcement prepared

### Communication
- [ ] Users notified of new release
- [ ] Changelog shared
- [ ] Known issues communicated
- [ ] Support channels ready

---

## Post-Release

### Monitoring
- [ ] Monitor download statistics
- [ ] Watch for bug reports
- [ ] Check support channels
- [ ] Review user feedback

### Backup
- [ ] Release artifacts backed up
- [ ] Source code tagged in git
- [ ] Build configuration saved
- [ ] Documentation archived

### Planning
- [ ] Next version planned
- [ ] Feature requests reviewed
- [ ] Bug fixes prioritized
- [ ] Roadmap updated

---

## Quick Reference

### Build Commands
```bash
# Clean previous builds
clean_build.bat

# Build release
build_release.bat

# Test release
test_release.bat

# Version management
python version.py                    # Show current version
python version.py bump patch         # Increment patch version
python version.py increment-build    # Increment build number
```

### File Locations
- Executable: `build_tools/dist/MyVoice/MyVoice.exe`
- Installer: `installer_output/MyVoice-v{version}/MyVoice-Setup-v{version}.exe`
- Checksums: `installer_output/MyVoice-v{version}/*.sha256.txt`

---

## Sign-Off

- [ ] **Release Manager:** _____________________________ Date: __________
- [ ] **QA Tester:** ______________________________________ Date: __________
- [ ] **Developer Lead:** ________________________________ Date: __________

---

**Version:** {version}
**Build Date:** {date}
**Release Type:** [ ] Major  [ ] Minor  [ ] Patch  [ ] Hotfix

---

*Complete all checkboxes before distributing the release*
