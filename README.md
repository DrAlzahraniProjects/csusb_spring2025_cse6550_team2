# CSUSB Study Abroad Chatbot

This repository will ultimately contain an AI chatbot able to answer queries pertaining to CSUSB's Study Abroad domain.

## Build Instructions
1. Install Docker.
2. Open a terminal of your choice, and run
```bash
docker build --tag "tag" .
```
To use less memory, you can optionally add `--no-cache` to the end of the command.
3. Once the image is built, run
```bash
docker run --publish 2502:2502 -it "<tag>"
```
This should run the application.
4. When finished, you can look up the running container's name via
```bash
docker ps -a
```
then stop and remove it, and the image, with
```bash
docker stop "<container name>"
docker rm "<container name>"
docker rmi "<tag>"
```