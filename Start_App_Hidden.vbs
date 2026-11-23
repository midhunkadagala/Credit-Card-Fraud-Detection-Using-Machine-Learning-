' Double-click this file to start the app with a minimal console window.
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sh.Run "cmd /c Start_Fraud_Detection_App.bat", 1, False
