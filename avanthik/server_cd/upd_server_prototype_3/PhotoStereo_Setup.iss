; ============================================================
; PhotoStereo Server — Windows Installer
; Built with Inno Setup 6.x  (https://jrsoftware.org/isinfo.php)
; ============================================================

#define AppName      "PhotoStereo Server"
#define AppVersion   "1.0.2"
#define AppPublisher "IndicVision"
#define InstallDir   "C:\PhotoStereo"
#define PythonExe    "python-3.11.9-amd64.exe"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={#InstallDir}
DefaultGroupName={#AppName}
OutputDir=Output
OutputBaseFilename=PhotoStereo_Setup
Compression=lzma2/ultra64
SolidCompression=yes

; Require admin rights so Python can be installed system-wide and firewall rules added
PrivilegesRequired=admin
WizardStyle=modern
UninstallDisplayName={#AppName}
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; ── Ensure generated files and sandboxes are wiped on uninstall ──
[UninstallDelete]
Type: filesandordirs; Name: "{#InstallDir}\env"
Type: filesandordirs; Name: "{#InstallDir}\__pycache__"
Type: filesandordirs; Name: "{%USERPROFILE}\photostereo_sessions"
Type: dirifempty; Name: "{#InstallDir}"

; ── What gets copied to C:\PhotoStereo\ ──────────────────────────────────────
[Files]
Source: "laptop_server.py";      DestDir: "{#InstallDir}"; Flags: ignoreversion
Source: "process_pipeline.py";   DestDir: "{#InstallDir}"; Flags: ignoreversion
Source: "masking.py";            DestDir: "{#InstallDir}"; Flags: ignoreversion
Source: "reconstruction_upd.py"; DestDir: "{#InstallDir}"; Flags: ignoreversion
Source: "gpu_installer.py";      DestDir: "{#InstallDir}"; Flags: ignoreversion
Source: "calibrate_lights.py";   DestDir: "{#InstallDir}"; Flags: ignoreversion

; The Python installer — bundled inside the Setup.exe, extracted temporarily
Source: "{#PythonExe}"; DestDir: "{tmp}"; Flags: deleteafterinstall

; ── Desktop shortcut — double-click to start the server ──────────────────────
[Icons]
Name: "{commondesktop}\Start PhotoStereo Server"; Filename: "{#InstallDir}\Start PhotoStereo Server.bat"; Comment: "Start the PhotoStereo laptop server"
Name: "{commondesktop}\Install NVIDIA GPU Acceleration"; Filename: "{#InstallDir}\Enable GPU Acceleration.bat"; Comment: "Download and install NVIDIA GPU support"
Name: "{commondesktop}\Run Calibration Standalone"; Filename: "{#InstallDir}\Run Calibration Standalone.bat"; Comment: "Compute light calibration factors from white paper images"

Name: "{group}\Start PhotoStereo Server"; Filename: "{#InstallDir}\Start PhotoStereo Server.bat"
Name: "{group}\Install NVIDIA GPU Acceleration"; Filename: "{#InstallDir}\Enable GPU Acceleration.bat"
Name: "{group}\Uninstall PhotoStereo Server"; Filename: "{uninstallexe}"

; ── Main install logic ────────────────────────────────────────────────────────
[Code]

function RunAndWait(Executable, Params, WorkDir: String): Integer;
var
  ResultCode: Integer;
begin
  Exec(Executable, Params, WorkDir, SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := ResultCode;
end;

function FindPython: String;
var
  ResultCode: Integer;
begin
  Result := '';
  if Exec('python', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      Result := 'python';
      Exit;
    end;
  end;
  if Exec('python3', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      Result := 'python3';
      Exit;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  PythonCmd: String;
  PythonInstaller: String;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin

    // ── STEP 1: Install Python if not present ──────────────────────────────
    WizardForm.StatusLabel.Caption := 'Checking for Python...';
    PythonCmd := FindPython();

    if PythonCmd = '' then
    begin
      WizardForm.StatusLabel.Caption :=
        'Installing Python 3.11 (this may take 1-2 minutes)...';
      PythonInstaller := ExpandConstant('{tmp}\{#PythonExe}');

      ResultCode := RunAndWait(PythonInstaller,
        '/quiet InstallAllUsers=1 PrependPath=1 Include_test=0',
        ExpandConstant('{tmp}'));
      if ResultCode <> 0 then
      begin
        MsgBox(
          'Python installation failed (exit code ' + IntToStr(ResultCode) + ').' + #13#10 +
          'Please install Python 3.11 manually from python.org' + #13#10 +
          'and make sure to tick "Add Python to PATH" during install.' + #13#10#13#10 +
          'Then re-run this installer.',
          mbError, MB_OK);
        Exit;
      end;
      PythonCmd := 'python';
    end;

    // ── STEP 2: Create Sandbox & Install Packages ─────────────────────────
    WizardForm.StatusLabel.Caption := 'Creating isolated Python sandbox...';
    RunAndWait(PythonCmd, '-m venv env', ExpandConstant('{#InstallDir}'));
    PythonCmd := ExpandConstant('{#InstallDir}\env\Scripts\python.exe');

    WizardForm.StatusLabel.Caption :=
      'Installing Python packages to sandbox (5-15 minutes)...' + #13#10 +
      'Please wait — do not close this window.';
    RunAndWait(PythonCmd, '-m pip install --upgrade pip --quiet', ExpandConstant('{#InstallDir}'));
      
    ResultCode := RunAndWait(PythonCmd,
      '-m pip install ' +
      'zeroconf ' +
      'rembg[cpu] ' +
      'opencv-contrib-python ' +
      'matplotlib ' +
      'numpy ' +
      'scipy ' +
      'pandas ' +
      'pyamg ' +
      'numba ' +
      'plotly ' +
      'psutil ' +
      '--quiet',
      ExpandConstant('{#InstallDir}'));
      
    if ResultCode <> 0 then
    begin
      MsgBox('Package installation failed. Check internet connection.', mbError, MB_OK);
      Exit; // <-- THE SAFETY CATCH
    end;

    // ── STEP 3: Add Windows Firewall rule for Python ───────────────────────
    WizardForm.StatusLabel.Caption := 'Configuring Windows Firewall...';
    RunAndWait('netsh',
      'advfirewall firewall add rule ' +
      'name="PhotoStereo Server" ' +
      'dir=in ' +
      'action=allow ' +
      'protocol=TCP ' +
      'localport=8080 ' +
      'program="' + PythonCmd + '" ' +
      'enable=yes ' +
      'profile=private,domain',
      '');

    // ── STEP 4: Create the launcher batch file ─────────────────────────────
    WizardForm.StatusLabel.Caption := 'Creating launchers...';
    SaveStringToFile(
      ExpandConstant('{#InstallDir}\Start PhotoStereo Server.bat'),
      '@echo off' + #13#10 +
      'title PhotoStereo Server' + #13#10 +
      'echo ============================================' + #13#10 +
      'echo   PhotoStereo Server' + #13#10 +
      'echo ============================================' + #13#10 +
      'echo.' + #13#10 +
      'echo  Make sure your laptop hotspot is ON' + #13#10 +
      'echo  and set to 2.4 GHz before continuing.' + #13#10 +
      'echo.' + #13#10 +
      'echo  Press any key to start the server...' + #13#10 +
      'pause > nul' + #13#10 +
      '"' + ExpandConstant('{#InstallDir}\env\Scripts\python.exe') + '" "' + ExpandConstant('{#InstallDir}\laptop_server.py') + '"' + #13#10 +
      'echo.' + #13#10 +
      'echo  Server stopped. Press any key to close.' + #13#10 +
      'pause > nul',
      False);

    // ── Create the GPU Upgrade batch file ──────────────────────────────────
    SaveStringToFile(
      ExpandConstant('{#InstallDir}\Enable GPU Acceleration.bat'),
      '@echo off' + #13#10 +
      'title NVIDIA GPU Setup' + #13#10 +
      '"' + ExpandConstant('{#InstallDir}\env\Scripts\python.exe') + '" "' + ExpandConstant('{#InstallDir}\gpu_installer.py') + '"' + #13#10,
      False);

    // ── Create the Calibration Standalone batch file ───────────────────────
    SaveStringToFile(
      ExpandConstant('{#InstallDir}\Run Calibration Standalone.bat'),
      '@echo off' + #13#10 +
      'title PhotoStereo Calibration' + #13#10 +
      'echo Running light calibration...' + #13#10 +
      'echo Make sure white paper images are in: %USERPROFILE%\photostereo_sessions\calibration\' + #13#10 +
      '"' + ExpandConstant('{#InstallDir}\env\Scripts\python.exe') + '" "' + ExpandConstant('{#InstallDir}\calibrate_lights.py') + '"' + #13#10 +
      'pause',
      False);

    WizardForm.StatusLabel.Caption := 'Installation complete!';

  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Exec('netsh',
      'advfirewall firewall delete rule name="PhotoStereo Server"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;