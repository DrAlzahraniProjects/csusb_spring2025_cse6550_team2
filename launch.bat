@echo off

set "APP_NAME=team2s25-app"
set "APP_PORT=2502"
set "NOTEBOOK_PORT=2582"

::Clean up old instances first
call "cleanup.bat"

::TODO: Check if port is open to inbound TCP traffic. Open it if not.
@REM netstat -ano | findstr %APP_PORT% 1>nul 2>nul
@REM if %ERRORLEVEL% NEQ 0 (
@REM 	netsh advfirewall firewall add rule name="CSUSB Travel Abroad Chatbot" dir=in action=allow protocol=TCP localport=%APP_PORT% 1>nul 2>nul
@REM 	if %ERRORLEVEL% NEQ 0 (
@REM 		echo Error: Failed to open port ^(error %ERRORLEVEL%^).
@REM 		exit \b %ERRORLEVEL%
@REM 	)
@REM 	cmd /c "exit 0"
@REM )

::Remove any containers already running on desired port
echo Vacating port %APP_PORT%...
docker ps -a -q --filter "publish=%APP_PORT%/tcp" | for /f %%i in ('findstr /r /v "^$"') do @docker stop %%i >nul 2>&1

::Create Docker image
echo Building app...
docker build -q -t "%APP_NAME%" . >nul 2>&1
@REM docker build -t "%APP_NAME%" .
if %ERRORLEVEL% NEQ 0 (
	echo Error: Failed to build Docker image ^(error %ERRORLEVEL%^).
	exit \b %ERRORLEVEL%
)
::Run Docker image
echo Launching app...
docker run -d -q --rm -p %APP_PORT%:%APP_PORT% -p %NOTEBOOK_PORT%:%NOTEBOOK_PORT% -it "%APP_NAME%" >nul 2>&1
@REM docker run -d --rm -p %APP_PORT%:%APP_PORT% -p %NOTEBOOK_PORT%:%NOTEBOOK_PORT% -it "%APP_NAME%"
if %ERRORLEVEL% == 0 (
	::Wait 5 seconds for the connection to reset; otherwise the user will be redirected to a webpage error
	timeout 5 /nobreak >nul 2>&1
	start "" http://localhost:%NOTEBOOK_PORT%/notebooks/notebook.ipynb >nul 2>&1
	start "" http://dev3:%APP_PORT% >nul 2>&1
) else (
	echo Error: Failed to run Docker image ^(error %ERRORLEVEL%^).
	exit \b %ERRORLEVEL%
)