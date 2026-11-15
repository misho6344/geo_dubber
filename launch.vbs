Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\user\Desktop\NoAdFrame\GeoDubber-Windows"
WshShell.Run "cmd /c start.bat", 0, False