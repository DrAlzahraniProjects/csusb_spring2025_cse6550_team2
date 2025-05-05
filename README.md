## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: Install [Git](https://git-scm.com/downloads) from the official webpage.
2. **Docker**: Install [Docker](https://www.docker.com/) from the official webpage.
3. **Linux/MacOS: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and [enable Docker's WSL integration](https://docs.docker.com/desktop/features/wsl/).

---

### Step 1: Delete the Existing Repository, if Present

```bash
rm -rf csusb_spring2025_cse6550_team2
```

### Step 2: Clone the Repository

Clone the GitHub repository to your local machine:

```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team2.git
```

### Step 3: Navigate to the Repository

Change to the cloned repository directory:

```bash
cd csusb_spring2025_cse6550_team2
```

### Step 4: Pull the Latest Version

Update the repository to the latest version:

```bash
git pull origin main
```

### Step 5: Enable Build Script to Run

This sets the launch and cleanup scripts to be executable.

```bash 
chmod +x cleanup.sh launch.sh
```

### Step 6: Run the Build Script (Enter your [Groq API key](https://console.groq.com/keys) when prompted):

```bash
./launch.sh
```

### Step 7: Access the Chatbot

When the container finishes building, the Streamlit app can be accessed locally at http://localhost:2502/team2s25 .

### Step 8: Run the Cleanup Script (This stops and removes the Docker image and container)

```bash
./cleanup.sh
```

---

### On the CSE Department Web Server:
- The Streamlit app is accessible remotely at https://sec.cse.csusb.edu/team2s25 .
### On Google Colab:
- A notebook version is accessible remotely at https://colab.research.google.com/drive/1Eb63IzbRTMMNWpYbvLjS2qjOwylJ6ogV .
