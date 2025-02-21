# CSUSB Study Abroad Chatbot

This repository will ultimately contain an AI chatbot able to answer queries pertaining to [CSUSB's Study Abroad domain](https://goabroad.csusb.edu/).

## Build Instructions
1. If not already possessing them, install [Docker](<https://www.docker.com/>) and [Git](https://git-scm.com/downloads). If using Windows, also install [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) and [enable Docker WSL integration](https://docs.docker.com/desktop/features/wsl/).
2. **Configure Docker to Run Without Sudo (Linux Users Only)**  
   **Important:** After installing Docker, Linux users must complete these steps to run Docker commands without `sudo`. This is required for the bash scripts to function properly.

   - **Create Docker Group** (if not already exists):
     ```bash
     sudo groupadd docker
     ```

   - **Add Your User to Docker Group**:
     ```bash
     sudo usermod -aG docker $USER
     ```

   - **Activate Group Changes**:
     - Log out and log back in, or restart your system.
     - Alternatively, use this command to refresh group membership without logout:
       ```bash
       newgrp docker
       ```

   - **Verify Non-Root Access**:
     ```bash
     docker run hello-world
     ```
     If successful, you'll see a confirmation message. If you get a "permission denied" error:
     - Restart Docker service:
       ```bash
       sudo systemctl restart docker
       ```
   **Documentation Reference**:  
   For more details or troubleshooting, see Docker's official guide:  
   [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

   **Note for Windows/Mac Users**:  
   This configuration is automatic in Docker Desktop. Only Linux users need these manual steps.

3. Visit https://console.groq.com/keys, create an account (or sign into an existing account), and create a new API key.
4. Open a<!--n administrative or elevated--> WSL instance on Windows, or a Bash terimnal on Linux.
5. Run the following command:
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
