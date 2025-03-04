# CSUSB Study Abroad Chatbot

This repository will ultimately contain an AI chatbot able to answer queries pertaining to [CSUSB's Study Abroad domain](https://goabroad.csusb.edu/).

## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/downloads) 
2. **Docker**: [Install Docker](https://www.docker.com/) 
3. **Linux/MacOS**: Configure Docker to not require `sudo` by following [this guide](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker's WSL integration by following [this guide](https://docs.docker.com/desktop/features/wsl/).

---

### Step 1: Clone the Repository

If a previous version of the repository exists, remove it first to ensure a clean setup:

```bash
rm -rf "csusb_spring2025_cse6550_team2"
```

Now, clone the GitHub repository to your local machine:

```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team2.git
```

### Step 2: Navigate to the Repository

Change to the cloned repository directory 
```bash
cd csusb_spring2025_cse6550_team2
```

### Step 3: Set execution permissions for necessary scripts:

```bash 
chmod +x cleanup.sh
chmod +x launch.sh
```


### Step 4: Run Build Script:

```bash
./launch.sh
```

- You will need a Groq key from [here](https://console.groq.com/keys)
- **Note:** The launch script may take upwards of 5-10 minutes on some systems; please have patience until it is finished.

### Step 5: Access the Chatbot

 Ultimately both the application and a google colab containing documentation will be launched on localhost:

For Streamlit: http://localhost:2502/team2s25


### Step 6: Run the Script to Stop and Remove the Docker Image and Container:

```bash
./cleanup.sh
```

---

### Hosted on CSE Department Web Server

For Streamlit: https://sec.cse.csusb.edu/team2s25

For Google Colab: (https://colab.research.google.com/drive/1Eb63IzbRTMMNWpYbvLjS2qjOwylJ6ogV)

---


### Questions This Chatbot Can and Cannot Answer

| Answerable | Unanswerable |
|------------|--------------|
| What study abroad programs are offered through CSUSB? | Is there a set schedule for the next Study Abroad 101 information sessions? |
| Are there specific scholarships for CSUSB students studying abroad? | What are the exact application deadlines for all study abroad programs at CSUSB? |
| How can I find partner universities for direct enrollment through CSUSB? | Can you provide a full list of partner universities that CSUSB has agreements with? |
| Does CSUSB provide assistance with obtaining a visa for studying abroad? | Does the CSUSB website have a list of study abroad scholarships from external organizations? |
| Can I study abroad in a country where English is not the primary language? | What is the internal deadline for the Fulbright Scholarship at CSUSB? |
