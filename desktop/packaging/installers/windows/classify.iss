; CLASSify Desktop — Windows Installer (Inno Setup)
;
; Compile with: iscc classify.iss
; Requires the PyInstaller bundle in ..\..\..\dist\CLASSify\

#define MyAppName "CLASSify Desktop"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "UK Center for Applied AI"
#define MyAppExeName "CLASSify.exe"
#define MyAppSource "..\..\..\..\dist\CLASSify"

[Setup]
AppId={{CLASSIFY-DESKTOP-001}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\CLASSify
DefaultGroupName=CLASSify
DisableProgramGroupPage=yes
OutputDir=..\..\..\..\dist\installers
OutputBaseFilename=CLASSify-Setup-{#MyAppVersion}-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Uncomment for code signing in CI:
; SignTool=signtool

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#MyAppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CLASSify Desktop"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall CLASSify Desktop"; Filename: "{uninstallexe}"
Name: "{autodesktop}\CLASSify Desktop"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch CLASSify Desktop"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
