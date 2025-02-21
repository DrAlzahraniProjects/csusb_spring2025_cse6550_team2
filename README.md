# CSUSB Study Abroad Chatbot

This repository will ultimately contain an AI chatbot able to answer queries pertaining to [CSUSB's Study Abroad domain](https://goabroad.csusb.edu/).

## Build Instructions
1. If not already possessing them, install [Docker](<https://www.docker.com/>) and [Git](https://git-scm.com/downloads). If using Windows, also install [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) and [enable Docker WSL integration](https://docs.docker.com/desktop/features/wsl/).
2. Visit https://console.groq.com/keys, create an account (or sign into an existing account), and create a new API key.
3. Open a<!--n administrative or elevated--> WSL instance on Windows, or a Bash terimnal on Linux.
4. Run the following command:
```bash
rm -rf "csusb_spring2025_cse6550_team2" ; git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team2.git ; cd "csusb_spring2025_cse6550_team2" ; chmod +x ./cleanup.sh ; chmod +x ./launch.sh ; ./launch.sh
```
This will clone the repository from Git; enter the newly-created folder; configure the launch and cleanup scripts to be able to execute; and run the launch script. Ultimately both the application and a Jupyter notebook containing documentation will be launched, both on localhost (http://localhost:2502/team2s25 and http://localhost:2502/team2s25/jupyter) and on an external server (https://sec.cse.csusb.edu/team2s25 and https://sec.cse.csusb.edu/team2s25/jupyter).

5. When finished, you can clean up the application and Jupyter notebook by running the cleanup script.
```bash
./cleanup.sh
```
|                       Answerable                                    | Unanswerable                                                                                    | 
|:-------------:                                                      |:--------------:                                                                                 |
| Does CSUSB offer study abroad programs?                             | Is there a set date for the Study Abroad 101 information sessions?                              | 
| Can I apply for a study abroad program at CSUSB?                    | Is the application deadline for Concordia University's summer semester available here?          | 
| Is Toronto a good place for students to live while studying abroad? | Does the chatbot provide a full list of CSUSB-approved direct enrollment universities?          |
| Do I need a visa to study at the University of Seoul?               | Does the chatbot list all available study abroad scholarships?                                  |
| Can I study in South Korea or Taiwan if I only know English?        | Is the internal deadline for the Fulbright Scholarship application set by CSUSB available here? |
