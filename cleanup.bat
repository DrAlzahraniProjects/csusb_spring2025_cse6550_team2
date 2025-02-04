@echo off
set "APP_NAME=team2s25-app"
@REM set "PORT=2502"

::Check if any Docker containers for "team2s25-app" exist. Remove them if so.
echo Cleaning up previous app instances...
docker ps -a -q --filter "ancestor=%APP_NAME%" | for /f %%i in ('findstr /r /v "^$"') do @docker rm -f %%i >nul 2>&1
::Check if any Docker images named "team2s25-app" exist. Remove them if so.
docker images -q %APP_NAME% | for /f %%i in ('findstr /r /v "^$"') do @docker rmi -f %%i >nul 2>&1

::TODO: Close port