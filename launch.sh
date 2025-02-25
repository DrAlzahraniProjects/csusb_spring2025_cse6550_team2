#!/bin/bash
# Define the team name and create the app name by appending "-app" to it.
TEAM_NAME="team2s25"
APP_NAME="$TEAM_NAME-app"

# Define the ports used by the app and the associated notebook.
APP_PORT=2502
NOTEBOOK_PORT=2512

# Define an array of URLs to be opened after launching the app.
# These include local addresses (using the defined ports) and remote addresses.
URLS=(\
	"https://sec.cse.csusb.edu/$TEAM_NAME/jupyter" \
	"https://sec.cse.csusb.edu/$TEAM_NAME" \
	"http://localhost:$NOTEBOOK_PORT/$TEAM_NAME/jupyter" \
	"http://localhost:$APP_PORT/$TEAM_NAME" \
)

# -----------------------------------------------------------------------------
# Detect the current operating system.
# Uses 'uname -s' and a case statement to set the OS variable.
# Reference: https://stackoverflow.com/a/3466183
# -----------------------------------------------------------------------------
case "$(uname -s)" in
	Linux*)   OS="linux";;       # Linux systems
	Darwin*)  OS="macintosh";;   # macOS systems
	CYGWIN*)  OS="cygwin";;      # Cygwin environment on Windows
	MINGW*)   OS="mingw";;       # Minimalist GNU for Windows
	MSYS_NT*) OS="msys";;        # MSYS on Windows
	*)        OS="unknown"       # Unknown OS
esac

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
# This prevents port conflicts by stopping containers bound to APP_PORT or NOTEBOOK_PORT.
# -----------------------------------------------------------------------------
echo "Vacating ports..."
docker ps -a -q --filter "publish=$APP_PORT/tcp" --filter "publish=$NOTEBOOK_PORT/tcp" | xargs -r docker stop > /dev/null 2>&1

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
echo "Building app..."
docker build -q -t "$APP_NAME" . > /dev/null 2>&1
if [ $? -ne 0 ]; then
	echo "Error: Failed to build Docker image (error $?)."
	exit $?
fi

# -----------------------------------------------------------------------------
# Launch the Docker container in detached mode.
# The container maps the specified APP_PORT and NOTEBOOK_PORT from the host to the container.
# The '--rm' flag ensures that the container is removed after it stops.
# -----------------------------------------------------------------------------
echo "Launching app..."
# TODO: Replace --env with docker build --secret
docker run -d -q --rm -p $APP_PORT:$APP_PORT -p $NOTEBOOK_PORT:$NOTEBOOK_PORT --env GROQ_API_KEY=$apiKey -it "$APP_NAME" > /dev/null 2>&1
if [ $? -eq 0 ]; then
	# Wait 5 seconds to allow the website to initialize, avoiding connection errors.
	sleep 5
	# Loop over each URL in the URLS array to open them in the browser.
	for url in "${URLS[@]}"; do
		# For macOS, use the 'open' command.
		if [ $OS == "macintosh" ]; then
			open $url
		else
			# For other OSes (e.g., Windows via WSL), set the BROWSER variable and use sensible-browser.
			export BROWSER="/mnt/c/Windows/explorer.exe"
			sensible-browser $url
		fi
	done
else
	# If the Docker container fails to launch, display an error and exit.
	echo "Error: Failed to run Docker image (error $?)."
	exit $?
fi
