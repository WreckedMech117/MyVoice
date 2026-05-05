Set WshShell = CreateObject("WScript.Shell")
ScriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run """" & ScriptDir & "\01_Run_MyVoice.bat""", 0, False
Set WshShell = Nothing
