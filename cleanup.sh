# !/bin/bash
TEAM_NAME="team2s25"
APP_NAME="$TEAM_NAME-app"

# Check if any Docker containers for "team2s25-app" exist. Remove them if so.
echo "Cleaning up previous app instances..."
docker ps -a -q --filter "ancestor=$APP_NAME" | xargs -r docker rm -f > /dev/null 2>&1
# Check if any Docker images named "team2s25-app" exist. Remove them if so.
docker images -q $APP_NAME | xargs -r docker rmi -f > /dev/null 2>&1

# TODO: Close browser tabs
# TODO: Close port