@echo off
echo Compilation de simple.py en simple.exe...
echo.

pyinstaller --onefile --clean simple.py

echo.
echo Compilation terminee!
echo L'executable se trouve dans le dossier "dist"
echo.

echo.
echo Termine! Fichiers crees:
echo - dist\simple.exe
pause
