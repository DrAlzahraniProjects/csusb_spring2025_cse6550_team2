# CSUSB Study Abroad Chatbot

This repository will ultimately contain an AI chatbot able to answer queries pertaining to CSUSB's Study Abroad domain.

## Build Instructions
1. Install Docker and Git.
2. Clone this repository using Git.
<!-- 3. Open the Dockerfile in an editor of your choice, and change `browser.serverAddress` in the Dockerfile's CMD command to your device's name. (On Windows, this is the `COMPUTERNAME` environment variable.) -->
3. Open port 2502 on your device.
4. Open a terminal of your choice, navigate to the clone's directory, and run
```bash
docker build --tag "tag" .
```
If memory is an issue, you can optionally add `--no-cache` to the end of the command.
5. Once the image is built, run
```bash
docker run --publish 2502:2502 -it "<tag>"
```
This should create a container that begins running the application.
6. When finished, you can look up the running container's name via
```bash
docker ps -a
```
then stop and remove it, and the image, with
```bash
docker stop "<container name>"
docker rm "<container name>"
docker rmi "<tag>"
```