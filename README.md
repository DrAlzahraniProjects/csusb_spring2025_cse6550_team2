# CSUSB Study Abroad Chatbot

This repository will ultimately contain an AI chatbot able to answer queries pertaining to [CSUSB's Study Abroad domain](https://goabroad.csusb.edu/).

## Build Instructions
1. If not already possessing them, install [Docker](<https://www.docker.com/>) and [Git](https://git-scm.com/downloads). If using Windows, also install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and [enable Docker WSL integration](https://docs.docker.com/desktop/features/wsl/).
2. Open a<!--n administrative or elevated--> terminal. Clone this repository using Git.
```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team2.git
```
It is also possible to download this repository as a compressed file (under the green Code button) and extract it, though this is not recommended.
<!-- 3. Open the Dockerfile in an editor of your choice, and change `browser.serverAddress` in the Dockerfile's CMD command to your device's name. (On Windows, this is the `COMPUTERNAME` environment variable.) -->
3. Navigate to the cloned/extracted folder.
```bash
cd "csusb_spring2025_cse6550_team2"
```
4. Run the launch script:
```bash
.\launch.bat
```
This should automatically launch the application.
5. When finished, you can clean up the application by running the cleanup script.
```bash
.\cleanup.bat
```

A Bash script is planned, but has not been created yet.