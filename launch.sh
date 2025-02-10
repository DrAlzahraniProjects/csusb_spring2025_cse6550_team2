# !/bin/bash
TEAM_NAME="team2s25"
APP_NAME="$TEAM_NAME-app"
APP_PORT=2502
NOTEBOOK_PORT=2512

APP_URL="http://localhost:$APP_PORT/$TEAM_NAME"
NOTEBOOK_URL="http://localhost:$NOTEBOOK_PORT/notebooks/notebook.ipynb"
# NOTEBOOK_URL="http://localhost:$NOTEBOOK_PORT/$TEAM_NAME/jupyter"

# Get current operating system
# From paxdiablo on Stack Overflow: https://stackoverflow.com/a/3466183
# In case we need more in future: https://en.wikipedia.org/wiki/Uname#Examples
case "$(uname -s)" in
	Linux*)   OS="linux";;
	Darwin*)  OS="macintosh";;
	CYGWIN*)  OS="cygwin";;
	MINGW*)   OS="mingw";;
	MSYS_NT*) OS="msys";;
	*)        OS="unknown"
esac

# Clean up old instances first
./"cleanup.sh"

# TODO: Check if port is open to inbound TCP traffic. Open it if not.
# netstat -ano | findstr $APP_PORT
# if [$? -ne 0]; then
# 	netsh advfirewall firewall add rule name="CSUSB Travel Abroad Chatbot" dir=in action=allow protocol=TCP localport=$APP_PORT
# 	if [$? -ne 0]; then
# 		echo "Error: Failed to open port \(error $?\)."
# 		exit $?
# 	fi
# 	cmd /c "exit 0" # TODO: Translate to Bash
# fi

# Remove any containers already running on desired port
echo "Vacating ports..."
docker ps -a -q --filter "publish=$APP_PORT/tcp" --filter "publish=$NOTEBOOK_PORT/tcp" | xargs -r docker stop > /dev/null 2>&1

# Create Docker image
echo "Building app..."
docker build -q -t "$APP_NAME" . > /dev/null 2>&1
if [ $? -ne 0 ]; then
	echo "Error: Failed to build Docker image \(error $?\)."
	exit $?
fi
# Run Docker image
echo "Launching app..."
docker run -d -q --rm -p $APP_PORT:$APP_PORT -p $NOTEBOOK_PORT:$NOTEBOOK_PORT -it "$APP_NAME" > /dev/null 2>&1
if [ $? -eq 0 ]; then
	# Wait 5 seconds for the website connections to initiate; otherwise the user will be redirected to a "Connection reset" error
	sleep 5
	if [ $OS == "macintosh" ]; then
		open $NOTEBOOK_URL
		open $APP_URL
	else
		export BROWSER="/mnt/c/Windows/explorer.exe"
		sensible-browser $NOTEBOOK_URL
		sensible-browser $APP_URL
	fi
else
	echo "Error: Failed to run Docker image \(error $?\)."
	exit $?
fi