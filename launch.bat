@echo off

set "APP_NAME=team2s25-app"
set "PORT=2502"

::Clean up old instances first
call "cleanup.bat"

::TODO: Check if port is open to inbound TCP traffic. Open it if not.
@REM netstat -ano | findstr %PORT% 1>nul 2>nul
@REM if %ERRORLEVEL% NEQ 0 (
@REM 	netsh advfirewall firewall add rule name="CSUSB Travel Abroad Chatbot" dir=in action=allow protocol=TCP localport=%PORT% 1>nul 2>nul
@REM 	if %ERRORLEVEL% NEQ 0 (
@REM 		echo Error: Failed to open port ^(error %ERRORLEVEL%^).
@REM 		exit \b %ERRORLEVEL%
@REM 	)
@REM 	cmd /c "exit 0"
@REM )

::Remove any containers already running on desired port
echo Vacating port %PORT%...
docker ps -a -q --filter "publish=%PORT%/tcp" | for /f %%i in ('findstr /r /v "^$"') do @docker rm -f %%i >nul 2>&1

::Create Docker image
echo Building app...
docker build -q -t "%APP_NAME%" . >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
	echo Error: Failed to build Docker image ^(error %ERRORLEVEL%^).
	exit \b %ERRORLEVEL%
)
::Run Docker image
echo Launching app...
docker run -d -q -p %PORT%:%PORT% -it "%APP_NAME%" >nul 2>&1
if %ERRORLEVEL% == 0 (
	::Wait a second for the connection to reset; otherwise the user will be redirected to a webpage error
	timeout 1 /nobreak >nul 2>&1
	start "" http://dev3:2502 >nul 2>&1
) else (
	echo Error: Failed to run Docker image ^(error %ERRORLEVEL%^).
	exit \b %ERRORLEVEL%
)