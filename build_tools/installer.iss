; MyVoice Inno Setup Installer Script
; Creates a professional Windows installer for MyVoice TTS Desktop Application
; Requires Inno Setup 6.x - Download from https://jrsoftware.org/isdl.php

; =============================================================================
; APPLICATION INFORMATION
; =============================================================================

#define MyAppName "MyVoice"
#define MyAppVersion "2.2.0"
#define MyAppBuild "56"
#define MyAppPublisher "MyVoice Development Team"
#define MyAppURL "https://github.com/myvoice/myvoice"
#define MyAppExeName "MyVoice.exe"
#define MyAppDescription "Desktop Text-to-Speech Application with Qwen3-TTS Voice Cloning, Emotion Control, and Voice Design"

; =============================================================================
; SETUP CONFIGURATION
; =============================================================================

[Setup]
; Application identity
AppId={{8F4B7C92-3D1E-4A5B-9C2E-7F8D4E6A1B3C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
AppCopyright=Copyright (C) 2025-2026 {#MyAppPublisher}
AppComments={#MyAppDescription}

; Installation directories
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableDirPage=no
DisableProgramGroupPage=no
AllowNoIcons=yes

; License and info files
LicenseFile=..\resources\license.txt
InfoBeforeFile=..\resources\readme.txt

; Output configuration
OutputDir=..\installer_output
OutputBaseFilename=MyVoice-Setup-v{#MyAppVersion}
SetupIconFile=..\src\icon\MyVoice.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; Compression
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMADictionarySize=1048576
LZMANumBlockThreads=2
DiskSpanning=no

; Visual appearance
WizardStyle=modern
WizardImageFile=..\src\icon\MyVoice_Splash_Installer.png
WizardSmallImageFile=..\src\icon\MyVoice_Small_Installer.png
WizardImageStretch=no
WizardImageAlphaFormat=defined

; Privileges and compatibility
; PrivilegesRequired=lowest installs per-user without a UAC prompt, which
; resolves two long-standing issues: (1) the default DirName resolves via
; {autopf} to %LOCALAPPDATA%\Programs\MyVoice (the Microsoft-recommended
; per-user app location, same pattern as VS Code/Slack/Discord) instead of
; %ProgramFiles%; (2) portable_paths.py's writes to {app}\config\,
; {app}\logs\, {app}\voice_files\ now succeed without elevation, which
; previously forced users to run-as-admin or relocate. Users who want a
; system-wide install can still elevate via the
; PrivilegesRequiredOverridesAllowed=dialog prompt. VB-Cable's installer
; (line 525-528) triggers its own UAC prompt independently of the outer
; setup, so the optional VB-Cable task still works at any privilege level.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
MinVersion=10.0.17763
ArchitecturesInstallIn64BitMode=x64compatible

; Uninstall configuration
UninstallDisplayName={#MyAppName}
UninstallFilesDir={app}\uninstall
CreateUninstallRegKey=yes

; Misc settings
DisableWelcomePage=no
DisableReadyPage=no
ShowLanguageDialog=auto
SetupLogging=yes

; =============================================================================
; LANGUAGES
; =============================================================================

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; =============================================================================
; TASKS (User Options)
; =============================================================================

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode
; Model Quality Selection - mutually exclusive radio buttons (Quality is default)
Name: "modelquality_quality"; Description: "Quality (1.7B) - Higher output quality (~3.4 GB VRAM, recommended)"; GroupDescription: "Qwen3-TTS Model Selection:"; Flags: exclusive
Name: "modelquality_small"; Description: "Small (0.6B) - Lower VRAM (~1.2 GB), faster inference"; GroupDescription: "Qwen3-TTS Model Selection:"; Flags: exclusive unchecked
; Optional components
Name: "vbcable"; Description: "Install VB-Audio Cable (required for microphone routing, ~1MB download, requires restart)"; GroupDescription: "Optional Components:"; Check: not IsVirtualCableInstalled

; =============================================================================
; FILES TO INSTALL
; =============================================================================

[Files]
; Main application executable
Source: "..\build_tools\dist\MyVoice\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; Application internals - all files including PyTorch, CUDA, Whisper, etc.
; Using recursesubdirs to include all subdirectories (~8000 files, ~2GB)
Source: "..\build_tools\dist\MyVoice\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; Documentation
Source: "..\resources\readme.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "..\resources\license.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

; NVIDIA CUDA Toolkit EULA — end-user-visible copy per NVIDIA EULA §1.1.2 #5
; (Story 18.5 AC #2). The bundled CUDA redistributable subset at
; _internal\cuda_redist\ is governed by this EULA; making the EULA visible at
; install root is the redistribution-terms-consistency requirement.
Source: "..\build_tools\dist\MyVoice\_internal\cuda_redist\EULA.txt"; DestDir: "{app}"; DestName: "NVIDIA_CUDA_EULA.txt"; Flags: ignoreversion

; Default voice files (ready-to-use samples)
Source: "..\src\install_files\default_voices\*"; DestDir: "{app}\voice_files"; Flags: ignoreversion createallsubdirs recursesubdirs

; NOTE: Quick Speak default profile is created automatically on first boot
; No need to ship a pre-configured CSV file

; VB-Cable installation scripts (copied to temp, deleted after install)
Source: "..\src\install_files\check_virtual_cable.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\src\install_files\download_vbcable.ps1"; DestDir: "{tmp}"; Flags: dontcopy; Tasks: vbcable
Source: "..\src\install_files\install_vbcable.ps1"; DestDir: "{tmp}"; Flags: dontcopy; Tasks: vbcable

; NOTE: Don't use "Flags: ignoreversion" on any shared system files
; NOTE: Legacy GPT-SoVITS files (7zr.exe, extract_7z.ps1, run_go_api_hidden.vbs) are no longer needed

; =============================================================================
; ICONS / SHORTCUTS
; =============================================================================

[Icons]
; Start Menu shortcuts
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyAppDescription}"
Name: "{group}\{cm:ProgramOnTheWeb,{#MyAppName}}"; Filename: "{#MyAppURL}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop shortcut (optional, based on user selection)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "{#MyAppDescription}"

; Quick Launch shortcut (optional, for Windows 7 and below)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

; =============================================================================
; REGISTRY ENTRIES
; =============================================================================

[Registry]
; Root: HKA (auto) writes to HKLM on admin installs and HKCU on per-user
; installs, matching the {auto*} directory-constants pattern. Hardcoding HKLM
; here breaks PrivilegesRequired=lowest with "Access is denied" (code 5)
; during the "Creating registry keys" step because per-user processes can't
; write to HKLM\Software. Windows resolves App Paths from both hives.
;
; Register application for proper uninstall
Root: HKA; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "Build"; ValueData: "{#MyAppBuild}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\{#MyAppPublisher}\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey

; Add to Windows "App Paths" for command-line access (optional)
Root: HKA; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"; Flags: uninsdeletekey

; =============================================================================
; RUN ACTIONS (Post-Installation)
; =============================================================================

[Run]
; Option to launch application after installation
; Only show if VB-Cable was not installed this session (restart required if installed)
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent; Check: ShouldAllowLaunch

; =============================================================================
; UNINSTALL ACTIONS
; =============================================================================

[UninstallDelete]
; Clean up any files created during runtime
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{localappdata}\{#MyAppName}"
Type: filesandordirs; Name: "{userappdata}\{#MyAppName}"

; =============================================================================
; CUSTOM MESSAGES
; =============================================================================

[Messages]
; Customize installation messages
WelcomeLabel1=Welcome to the [name] Setup Wizard
WelcomeLabel2=This will install [name/ver] on your computer.%n%nMyVoice is a desktop text-to-speech application with embedded Qwen3-TTS for high-quality voice cloning, emotion control, and voice design.%n%nIMPORTANT: This installation requires approximately 3 GB of disk space PLUS the TTS model files (~1.2 GB for Small tier, ~3.4 GB for Quality tier). The model files are downloaded during installation so your first generation is instant — an internet connection is required.%n%nPlease be patient during the extraction and download process.%n%nIt is recommended that you close all other applications before continuing.
FinishedHeadingLabel=Completing the [name] Setup Wizard
FinishedLabelNoIcons=Setup has finished installing [name] on your computer.
FinishedLabel=Setup has finished installing [name] on your computer. The application may be launched by selecting the installed shortcuts.

[CustomMessages]
; Custom message for VB-Cable restart requirement
RestartRequiredForVBCable=IMPORTANT: Restart Required%n%nVB-Audio Cable was installed during setup. You must restart your computer before the virtual audio driver will function properly.%n%nThe microphone routing feature will not work until after restart.

; =============================================================================
; PASCAL SCRIPT (Advanced Features)
; =============================================================================

[Code]
var
  VBCableInstalledThisSession: Boolean;
  VirtualCableCheckDone: Boolean;
  VirtualCableFound: Boolean;

// Forward declarations
function IsVirtualCableInstalled(): Boolean; forward;
function ShouldAllowLaunch(): Boolean; forward;
procedure RunVirtualCableCheck(); forward;
procedure WriteModelQualitySettings(); forward;
procedure PredownloadModels(); forward;

// Perform the actual VB-Cable detection with progress indicator
procedure RunVirtualCableCheck();
var
  ResultCode: Integer;
  CheckScript: String;
  ProgressPage: TOutputProgressWizardPage;
begin
  if VirtualCableCheckDone then
    Exit;  // Already checked

  // Create and show progress page
  ProgressPage := CreateOutputProgressPage('Checking Audio Configuration',
    'Detecting virtual audio devices...');
  ProgressPage.SetText('Scanning audio drivers...', 'This may take a few moments.');
  ProgressPage.SetProgress(50, 100);
  ProgressPage.Show;

  try
    CheckScript := ExpandConstant('{tmp}\check_virtual_cable.ps1');

    // Extract check script
    ExtractTemporaryFile('check_virtual_cable.ps1');

    // Run detection script
    if Exec('powershell.exe',
      '-ExecutionPolicy Bypass -NoProfile -File "' + CheckScript + '"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      // Exit code 0 = found, 1 = not found
      VirtualCableFound := (ResultCode = 0);
      if VirtualCableFound then
        Log('Virtual cable detected')
      else
        Log('Virtual cable not detected');
    end
    else
    begin
      Log('Failed to run virtual cable check script');
      VirtualCableFound := False;
    end;

    VirtualCableCheckDone := True;
    ProgressPage.SetProgress(100, 100);
  finally
    ProgressPage.Hide;
  end;
end;

// Check if VB-Audio Virtual Cable is already installed (uses cached result)
function IsVirtualCableInstalled(): Boolean;
begin
  // Use cached result from earlier check
  if VirtualCableCheckDone then
  begin
    Result := VirtualCableFound;
  end
  else
  begin
    // Fallback: should not happen, but run check if needed
    RunVirtualCableCheck();
    Result := VirtualCableFound;
  end;
end;

// Check if Visual C++ Redistributable is installed (if needed)
function VCRedistNeedsInstall: Boolean;
var
  Version: String;
begin
  // Check if VC++ 2015-2022 Redistributable is installed
  // Most systems have this, but check to be safe
  Result := not RegQueryStringValue(HKLM,
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
    'Version', Version);
end;

// Show custom progress during file extraction
procedure InitializeWizard;
begin
  // Initialize tracking variables
  VBCableInstalledThisSession := False;
  VirtualCableCheckDone := False;
  VirtualCableFound := False;

  // Note: We use the built-in progress indicator instead of a custom ProgressPage
  // The built-in progress updates automatically during file extraction
  // This provides better user feedback for the ~2GB / 8000+ file installation
end;

// Check if MyVoice should be allowed to launch after installation
// Returns False if VB-Cable was just installed (restart required)
function ShouldAllowLaunch(): Boolean;
begin
  Result := not VBCableInstalledThisSession;
  if not Result then
    Log('Launch disabled: VB-Cable was installed this session, restart required');
end;

// Write initial settings.json with user-selected model quality tier
procedure WriteModelQualitySettings();
var
  ConfigDir: String;
  SettingsFile: String;
  QualityTier: String;
  SettingsContent: String;
begin
  // Determine which model quality was selected
  if WizardIsTaskSelected('modelquality_small') then
    QualityTier := 'small'
  else
    QualityTier := 'quality';  // Default to quality

  Log('Selected model quality tier: ' + QualityTier);

  // Create config directory
  ConfigDir := ExpandConstant('{app}\config');
  if not DirExists(ConfigDir) then
  begin
    if CreateDir(ConfigDir) then
      Log('Created config directory: ' + ConfigDir)
    else
    begin
      Log('Failed to create config directory: ' + ConfigDir);
      Exit;
    end;
  end;

  // Create settings.json with model quality tier
  SettingsFile := ConfigDir + '\settings.json';

  // Only create if it doesn't exist (don't overwrite user settings on reinstall)
  if not FileExists(SettingsFile) then
  begin
    // Write minimal settings.json with model quality tier
    // The application will merge defaults on first load
    SettingsContent := '{' + #13#10 +
      '  "model_quality_tier": "' + QualityTier + '"' + #13#10 +
      '}';

    if SaveStringToFile(SettingsFile, SettingsContent, False) then
      Log('Created settings.json with model_quality_tier: ' + QualityTier)
    else
      Log('Failed to create settings.json');
  end
  else
  begin
    Log('settings.json already exists, preserving user settings');
  end;
end;

// Pre-download the selected tier's TTS models so the first generation
// doesn't trigger a multi-GB HuggingFace download. Routed to
// `MyVoice.exe --predownload-models --tier=<small|quality>` which exits
// BEFORE the torch + PyQt6 imports run (see src\myvoice\main.py).
// Non-fatal on failure: if the network drops or HF is unreachable, we
// warn the user but allow install to complete — the app will fall back
// to downloading on first launch.
procedure PredownloadModels();
var
  ResultCode: Integer;
  QualityTier: String;
  PredownloadProgressPage: TOutputProgressWizardPage;
  ModelExeName: String;
  CmdLine: String;
  EstimatedSize: String;
begin
  // Determine which tier was selected (mirrors WriteModelQualitySettings)
  if WizardIsTaskSelected('modelquality_small') then
  begin
    QualityTier := 'small';
    EstimatedSize := '~1.2 GB';
  end
  else
  begin
    QualityTier := 'quality';
    EstimatedSize := '~3.4 GB';
  end;

  Log('Pre-downloading TTS models for tier: ' + QualityTier);

  PredownloadProgressPage := CreateOutputProgressPage(
    'Downloading TTS Models',
    'Pre-downloading the selected Qwen3-TTS model so the first ' +
    'generation is instant. This requires an internet connection.');
  PredownloadProgressPage.SetText(
    'Downloading model files (' + EstimatedSize + ')...',
    'This may take several minutes depending on your connection. ' +
    'The window will not appear to update during the download; ' +
    'please wait for it to complete.');
  PredownloadProgressPage.SetProgress(20, 100);
  PredownloadProgressPage.Show;

  try
    ModelExeName := ExpandConstant('{app}\{#MyAppExeName}');
    CmdLine := '--predownload-models --tier=' + QualityTier;

    if Exec(ModelExeName, CmdLine, '', SW_HIDE,
            ewWaitUntilTerminated, ResultCode) then
    begin
      PredownloadProgressPage.SetProgress(100, 100);
      if ResultCode = 0 then
      begin
        Log('Model pre-download successful (tier: ' + QualityTier + ')');
      end
      else if ResultCode = 2 then
      begin
        Log('Model pre-download partially failed (exit code 2). ' +
            'Remaining models will be fetched on first launch.');
        MsgBox(
          'Some TTS model files could not be downloaded. The ' +
          'application will fetch the missing files automatically on ' +
          'first generation.' + #13#10 + #13#10 +
          'Most common cause: temporary network interruption. ' +
          'This is not a fatal error — installation will complete normally.',
          mbInformation, MB_OK);
      end
      else
      begin
        Log('Model pre-download failed with code: ' + IntToStr(ResultCode));
        MsgBox(
          'TTS model pre-download failed (exit code: ' +
          IntToStr(ResultCode) + ').' + #13#10 + #13#10 +
          'The application will download the model on first launch instead. ' +
          'Installation will continue normally.',
          mbInformation, MB_OK);
      end;
    end
    else
    begin
      Log('Failed to launch MyVoice.exe for model pre-download');
      MsgBox(
        'Could not start the model pre-download. ' +
        'The application will download the model on first launch instead.' + #13#10 + #13#10 +
        'Installation will continue normally.',
        mbInformation, MB_OK);
    end;
  finally
    PredownloadProgressPage.Hide;
  end;
end;

// Handle optional component installation after file extraction
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  TempDownloadScript: String;
  TempInstallScript: String;
  VBCableProgressPage: TOutputProgressWizardPage;
begin
  // Track installation start time
  if CurStep = ssInstall then
  begin
    Log('Starting file installation...');
    // Built-in progress indicator handles file extraction progress automatically
  end;

  // ============================================================================
  // Post-installation: Write settings and install optional components
  // ============================================================================
  if CurStep = ssPostInstall then
  begin
    Log('File installation complete.');

    // ============================================================================
    // Write initial settings.json with selected model quality tier
    // ============================================================================
    WriteModelQualitySettings();

    // ============================================================================
    // Pre-download the selected tier's TTS models into the HuggingFace cache
    // so the first user-facing generation doesn't have to wait for the
    // multi-GB download. UAC elevation runs in the same user session, so
    // the cache populated here is the same one the runtime app reads from.
    // ============================================================================
    PredownloadModels();

    if WizardIsTaskSelected('vbcable') then
    begin
      // Create progress page just for VB-Cable installation
      VBCableProgressPage := CreateOutputProgressPage('Installing VB-Audio Cable',
        'Please wait while the virtual audio driver is downloaded and installed...');
      VBCableProgressPage.SetProgress(0, 100);
      VBCableProgressPage.Show;

      try
        ExtractTemporaryFile('download_vbcable.ps1');
        ExtractTemporaryFile('install_vbcable.ps1');

        TempDownloadScript := ExpandConstant('{tmp}\download_vbcable.ps1');
        TempInstallScript := ExpandConstant('{tmp}\install_vbcable.ps1');

        // Download VB-Cable
        VBCableProgressPage.SetText('Downloading VB-Audio Cable...', 'This requires an internet connection (~1MB download)');
        VBCableProgressPage.SetProgress(10, 100);
        Log('Downloading VB-Audio Cable...');

        if Exec('powershell.exe',
               '-ExecutionPolicy Bypass -NoProfile -File "' + TempDownloadScript + '"',
               '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
        begin
          if ResultCode = 0 then
          begin
            Log('VB-Cable download successful');
            VBCableProgressPage.SetText('Installing VB-Audio Cable...', 'Administrator approval may be required...');
            VBCableProgressPage.SetProgress(50, 100);

            // Install VB-Cable (will show UAC prompt)
            if Exec('powershell.exe',
                   '-ExecutionPolicy Bypass -NoProfile -File "' + TempInstallScript + '"',
                   '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
            begin
              VBCableProgressPage.SetProgress(100, 100);

              if ResultCode = 0 then
              begin
                Log('VB-Cable installation successful');
                VBCableInstalledThisSession := True;
                MsgBox('VB-Audio Cable installed successfully.' + #13#10 + #13#10 +
                       'IMPORTANT: You must restart your PC for the virtual audio cable to work.' + #13#10 +
                       'The microphone routing feature will not function until after restart.',
                       mbInformation, MB_OK);
              end
              else
              begin
                Log('VB-Cable installation failed with code: ' + IntToStr(ResultCode));
                MsgBox('VB-Audio Cable installation failed (exit code: ' + IntToStr(ResultCode) + ').' + #13#10 + #13#10 +
                       'You can install it manually later from:' + #13#10 +
                       'https://vb-audio.com/Cable/', mbError, MB_OK);
              end;
            end
            else
            begin
              Log('Failed to execute VB-Cable installer script');
              MsgBox('Failed to run VB-Cable installer. You can install it manually later from:' + #13#10 +
                     'https://vb-audio.com/Cable/', mbError, MB_OK);
            end;
          end
          else
          begin
            Log('VB-Cable download failed with code: ' + IntToStr(ResultCode));
            MsgBox('VB-Audio Cable download failed (exit code: ' + IntToStr(ResultCode) + ').' + #13#10 + #13#10 +
                   'This may be due to network issues. You can download and install it manually from:' + #13#10 +
                   'https://vb-audio.com/Cable/', mbError, MB_OK);
          end;
        end
        else
        begin
          Log('Failed to execute VB-Cable download script');
          MsgBox('Failed to start VB-Cable download. You can download and install it manually from:' + #13#10 +
                 'https://vb-audio.com/Cable/', mbError, MB_OK);
        end;
      finally
        VBCableProgressPage.Hide;
      end;
    end;
  end;
end;

// Clean up on setup exit
procedure DeinitializeSetup;
begin
  // Nothing to clean up - using local progress pages
  Log('Setup cleanup complete.');
end;

// Check for minimum disk space before installation
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  FreeMB: Cardinal;
  RequiredMB: Cardinal;
begin
  Result := '';
  RequiredMB := 3000;  // ~3GB required for installation

  // Check available disk space on target drive
  // GetSpaceOnDisk returns space in MB
  if not GetSpaceOnDisk(ExpandConstant('{app}'), True, FreeMB, FreeMB) then
  begin
    Log('Could not check disk space, proceeding anyway');
  end
  else
  begin
    Log('Available disk space: ' + IntToStr(FreeMB) + ' MB');
    if FreeMB < RequiredMB then
    begin
      Result := 'Insufficient disk space.' + #13#10 + #13#10 +
                'MyVoice requires at least ' + IntToStr(RequiredMB) + ' MB of free space.' + #13#10 +
                'Available: ' + IntToStr(FreeMB) + ' MB' + #13#10 + #13#10 +
                'Please free up disk space and try again.';
    end;
  end;
end;

// Run VB-Cable detection when user clicks Next on Program Group page
// This runs BEFORE the Tasks page is displayed, with a progress indicator
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;  // Allow navigation by default

  // When leaving the Start Menu Folder page (before Tasks page)
  if CurPageID = wpSelectProgramGroup then
  begin
    if not VirtualCableCheckDone then
    begin
      Log('Running VB-Cable detection before Tasks page...');
      RunVirtualCableCheck();
    end;
  end;
end;

// Handle page changes for post-installation actions
procedure CurPageChanged(CurPageID: Integer);
begin
  // Handle post-installation restart requirement for VB-Cable
  if CurPageID = wpFinished then
  begin
    if VBCableInstalledThisSession then
    begin
      Log('VB-Cable was installed - restart is strongly recommended');
      // Show prominent restart requirement message
      MsgBox(ExpandConstant('{cm:RestartRequiredForVBCable}'), mbInformation, MB_OK);
      // Note: The launch option is already hidden via ShouldAllowLaunch check
    end;
  end;
end;
