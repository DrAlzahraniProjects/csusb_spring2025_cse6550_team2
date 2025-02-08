# !/bin/bash
APP_NAME="team2s25-app"
APP_PORT=2502
NOTEBOOK_PORT=2582

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
echo "Vacating port $APP_PORT..."
docker ps -a -q --filter "publish=$APP_PORT/tcp" | xargs -r docker stop > /dev/null 2>&1

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
	export BROWSER="/mnt/c/Windows/explorer.exe"
	sensible-browser http://127.0.0.1:$NOTEBOOK_PORT/notebooks/notebook.ipynb
	sensible-browser http://127.0.0.1:$APP_PORT
else
	echo "Error: Failed to run Docker image \(error $?\)."
	exit $?
fi
