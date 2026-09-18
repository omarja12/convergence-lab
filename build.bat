@echo off
REM Build without CMake using MSVC.
REM
REM Run from a Developer Command Prompt and this goes straight to the compiler.
REM Otherwise it locates vcvars64.bat itself, via vswhere if present and a short
REM list of default install paths if not.
setlocal

if defined VSCMD_ARG_TGT_ARCH goto :compile
if defined VCINSTALLDIR goto :compile

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :try_defaults
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath 2^>nul`) do set "VSPATH=%%i"
if not defined VSPATH goto :try_defaults
if exist "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" goto :have_vcvars

:try_defaults
for %%p in (
  "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Community"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Professional"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools"
) do if exist "%%~p\VC\Auxiliary\Build\vcvars64.bat" call :set_path "%%~p"
if defined VSPATH goto :have_vcvars

echo.
echo Could not locate the MSVC build tools.
echo Install "Desktop development with C++", or run this from a
echo Developer Command Prompt, or use CMake instead:
echo     cmake -B build -DCMAKE_BUILD_TYPE=Release ^&^& cmake --build build
exit /b 1

:set_path
if not defined VSPATH set "VSPATH=%~1"
goto :eof

:have_vcvars
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

:compile
if not exist build mkdir build
set "FLAGS=/nologo /std:c++20 /EHsc /O2 /W4 /Zc:__cplusplus /I include"
cl %FLAGS% src\main.cpp /Fe:build\convergence_demo.exe /Fo:build\ || exit /b 1
cl %FLAGS% tests\test_convergence.cpp /Fe:build\convergence_tests.exe /Fo:build\ || exit /b 1
echo.
echo Built build\convergence_demo.exe and build\convergence_tests.exe
exit /b 0
