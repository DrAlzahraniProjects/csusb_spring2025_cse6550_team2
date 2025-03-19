#!/bin/bash
# Define the team name and create the app name by appending "-app" to it.
TEAM_NAME="team2s25"
APP_NAME="$TEAM_NAME-app"

# Define the ports used by the app and the associated notebook.
APP_PORT=2502

# -----------------------------------------------------------------------------
# Detect the current operating system.
# Uses 'uname -s' and a case statement to set the OS variable.
# Reference: https://stackoverflow.com/a/3466183
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Clean up old Docker instances by running the cleanup script.
# This ensures that previous instances are stopped before launching a new one.
# -----------------------------------------------------------------------------
./"cleanup.sh"

# -----------------------------------------------------------------------------
# TODO: Check if the required port is open for inbound TCP traffic.
# The following commented-out code shows one approach (using netstat and netsh)
# to open the port if it is not open. Adjust as necessary.
#
# netstat -ano | findstr $APP_PORT
# if [ $? -ne 0 ]; then
# 	netsh advfirewall firewall add rule name="CSUSB Travel Abroad Chatbot" dir=in action=allow protocol=TCP localport=$APP_PORT
# 	if [ $? -ne 0 ]; then
# 		echo "Error: Failed to open port \(error $?\)."
# 		exit $?
# 	fi
# 	cmd /c "exit 0" # TODO: Translate to Bash if needed
# fi
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Remove any Docker containers that are already using the desired ports.
# -----------------------------------------------------------------------------
echo "Vacating ports..."
docker ps -a -q --filter "publish=$APP_PORT/tcp" | xargs -r docker stop > /dev/null 2>&1

apiKey=""
echo "------------------------------------------------------------------------------------------------------"
echo "This app requires a Groq API key to operate."
echo "(If you don't have one, visit https://console.groq.com/keys and sign in to your account/create a new account, then generate a new API key.)"
read -r -p "Key: " apiKey
echo "------------------------------------------------------------------------------------------------------"

# -----------------------------------------------------------------------------
# Build the Docker image for the app.
# The '-q' flag ensures that only essential output is shown.
# -----------------------------------------------------------------------------
echo "Building app... (Warning: may take 3-8 minutes)"
docker build -q -t "$APP_NAME" . > /dev/null 2>&1
# docker build -t "$APP_NAME" .
if [ $? -ne 0 ]; then
	echo "Error: Failed to build Docker image (error $?)."
	exit $?
fi

# -----------------------------------------------------------------------------
# Launch the Docker container in detached mode.
# The '--rm' flag ensures that the container is removed after it stops.
# -----------------------------------------------------------------------------
echo "Launching app..."
# TODO: Replace --env with docker build --secret
docker run -d -q --rm -p $APP_PORT:$APP_PORT --env GROQ_API_KEY=$apiKey -it "$APP_NAME" --name "$APP_NAME" > /dev/null 2>&1
if [ $? -eq 0 ]; then
	# Output where the apps are running
echo "Streamlit is available at: http://localhost:$APP_PORT/$TEAM_NAME"
echo "https://colab.research.google.com/drive/1Eb63IzbRTMMNWpYbvLjS2qjOwylJ6ogV"
else
	# If the Docker container fails to launch, display an error and exit.
	echo "Error: Failed to run Docker image (error $?)."
	exit $?
fi