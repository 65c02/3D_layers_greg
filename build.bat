@echo off
echo Compilation de simple.py en simple.exe...
echo.

pyinstaller --onefile --clean simple.py

echo.
echo Compilation terminee!
echo L'executable se trouve dans le dossier "dist"
echo.

echo Creation du fichier simple.zip...
if exist simple.zip del simple.zip
powershell Compress-Archive -Path dist\simple.exe -DestinationPath simple.zip -Force

echo.
echo Termine! Fichiers crees:
echo - dist\simple.exe
echo - simple.zip
pause
