# Import necessary libraries
from langchain_groq import ChatGroq  # Library for interacting with the Groq API
import os                           # Provides functions to interact with the operating system
import streamlit as st              # Library for building web apps
import time                         # Provides time-related functions

# ==============================================================================================================================
# WARNING:
# Streamlit apparently has the horrific property of RERUNNING THE MAJORITY OF THE CODE, INCLUDING GLOBAL VARIABLE DECLARATIONS,
# EVERY TIME IT REACHES THE END OF THIS FILE (currently, after each pair of human + AI messages).
# To have (non-constant) variables persist beyond that, you must instead store their values in st.session_state["..."] instead.
# ==============================================================================================================================

# Constants for cooldown and response management
COOLDOWN_CHECK_PERIOD: float = 60.          # Time period (in seconds) to check the number of messages sent
MAX_MESSAGES_BEFORE_COOLDOWN: int = 10        # Maximum messages allowed within the cooldown check period
COOLDOWN_DURATION: float = 180.               # Duration (in seconds) for which the app is in cooldown if the limit is exceeded
MAX_RESPONSE_TIME: float = 3.                 # Threshold for response time (in seconds) to flag slow responses
SHOWN_API_KEY_CHARACTERS_COUNT: int = 8       # Number of characters to show when displaying the API key

# System prompt that defines the behavior and rules for the AI chatbot
SYSTEM_PROMPT: str = """
You are an expert assistant for the Study Abroad program of California State University, San Bernardino (CSUSB). You are designed to help students with all questions related to studying abroad.
You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.
You remain professional, encouraging, and optimistic at all times, ensuring students feel supported and motivated to pursue their dreams of studying overseas.
Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student. If a question is unrelated (e.g. politics or unrelated personal advice), politely guide the user back to study abroad topics.
- **No Negative Responses:** While you must remain truthful at all times, also avoid negative opinions, discouragement, or any response that may deter students from studying abroad.
- **Encourage and Inform:** Provide factual, detailed, and encouraging responses to all study abroad inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad, including politics, religion, or personal debates.
"""

# -----------------------------------------------------------------------------
# Function: canAnswer
# Purpose: Check if the user can send a new message based on the cooldown logic.
# Returns: True if a new message is allowed; otherwise, prints an error with remaining cooldown time and returns False.
# -----------------------------------------------------------------------------
def canAnswer() -> bool:
    # Get the current monotonic timestamp (monotonic clock ensures it only goes forward)
    currentTimestamp = time.monotonic()
    
    # Check if a cooldown period is already in effect
    if st.session_state["cooldownBeginTimestamp"] is not None:
        # If the cooldown period has passed, reset it and allow answering
        if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
    else:
        # Not in cooldown: update the list of recent message timestamps (keep only the most recent ones)
        st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:]
        st.session_state["messageTimes"].append(currentTimestamp)
        # Check if the number of messages is within the allowed limit or if they were spread out over time
        if len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or \
           st.session_state["messageTimes"][-1] - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD:
            return True
        # If the limit is exceeded, start the cooldown period
        else:
            st.session_state["cooldownBeginTimestamp"] = currentTimestamp
    
    # Calculate the remaining cooldown time for user feedback
    cooldownMinutes = int(COOLDOWN_CHECK_PERIOD // 60)
    cooldownSeconds = int(COOLDOWN_CHECK_PERIOD) % 60
    remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
    remainingMinutes = int(remainingTime // 60)
    remainingSeconds = int(remainingTime) % 60
    
    # Inform the user about the cooldown and remaining wait time
    st.write(f"ERROR: You've reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} question{'' if MAX_MESSAGES_BEFORE_COOLDOWN == 1 else 's'} per {cooldownMinutes} minute{'' if cooldownMinutes == 1 else 's'}{' ' + str(cooldownSeconds) + ' second' + ('' if cooldownSeconds == 1 else 's') if cooldownSeconds else ''} because the server has limited resources. Please try again in {remainingMinutes} minute{'' if remainingMinutes == 1 else 's'}{' ' + str(remainingSeconds) + ' second' + ('' if remainingSeconds == 1 else 's') if remainingSeconds else ''}.")
    
    return False


# -----------------------------------------------------------------------------
# Function: apiBox
# Purpose: Display and update the API key input box in the Streamlit app.
# -----------------------------------------------------------------------------
def apiBox():
    # If the API key is not set in the environment but exists in session_state, update the environment variable.
    if ("GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]) and "GROQ_API_KEY" in st.session_state:
        os.environ["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"]
    
    # Create a container for the API key input elements with a border
    with st.container(border=True):
        # Calculate half the characters to be displayed for masking the API key
        halfCharactersCount = int(SHOWN_API_KEY_CHARACTERS_COUNT // 2)
        
        # Provide a text input for entering a new API key (input is hidden as it's a password)
        newAPIkey = st.text_input("New Groq API key:", placeholder="[New Grok API key]", type="password", label_visibility="hidden")
        
        # Update the API key in both session_state and the environment variable
        st.session_state["GROQ_API_KEY"] = os.environ["GROQ_API_KEY"] = newAPIkey
        
        # Display the current API key in a partially masked form for security
        st.write(
            f"Your current API key is {'[none provided]' if 'GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY'] else os.environ['GROQ_API_KEY'] if len(os.environ['GROQ_API_KEY']) <= SHOWN_API_KEY_CHARACTERS_COUNT else os.environ['GROQ_API_KEY'][:halfCharactersCount] + '...' + os.environ['GROQ_API_KEY'][halfCharactersCount - SHOWN_API_KEY_CHARACTERS_COUNT:]}. Type a new key in the field above to change."
        )
        # Uncomment the following line if you want to force a rerun when the API key is updated.
        # st.rerun()


# -----------------------------------------------------------------------------
# Function: mainPage
# Purpose: Render the main page of the chatbot application.
# -----------------------------------------------------------------------------
def mainPage() -> None:
    # Display the main title for the chatbot application
    st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")
    
    # Render the API key input box at the top of the page
    apiBox()
    
    # Initialize session state variables if they haven't been set yet.
    if "cooldownBeginTimestamp" not in st.session_state:
        st.session_state["cooldownBeginTimestamp"] = None
    if "messages" not in st.session_state or not isinstance(st.session_state["messages"], list):
        st.session_state["messages"] = []
    if "messageTimes" not in st.session_state or not isinstance(st.session_state["messageTimes"], list):
        st.session_state["messageTimes"] = []
    
    # Display previous chat messages (both human and AI) from session_state
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Create an instance of the ChatGroq AI with specified parameters
    ai = ChatGroq(
        model="llama-3.1-8b-instant",  # Specify the model name
        temperature=0,                 # Deterministic output (no randomness)
        max_tokens=None,               # No limit on the number of tokens
        timeout=None,                  # No timeout set
        max_retries=2,                 # Retry twice if an error occurs
        # Other parameters can be added here...
    )
    
    # Variables to record the response time of the AI
    responseStartTime, responseEndTime = 0., 0.
    
    # Create a chat input box. If no API key is provided, disable the input.
    prompt = st.chat_input(
        "This app cannot be used without an API key." if 'GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY'] else "What is your question?",
        disabled='GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY']
    )
    
    # If the user has submitted a prompt and is allowed to send a message (not in cooldown)
    if prompt and canAnswer():
        # Display the user's message in the chat window
        st.chat_message("human").markdown(prompt)
        # Save the user's message in the session_state message list
        st.session_state["messages"].append({"role": "human", "content": prompt})
        
        # Prepare the message history for the AI, including the system prompt and previous conversation messages
        messages = [("system", SYSTEM_PROMPT)] + [(message["role"], message["content"]) for message in st.session_state["messages"]]
        
        # Record the start time for measuring response time
        responseStartTime = time.monotonic()
        
        # Process and display the AI response in the chat interface
        with st.chat_message("ai"):
            # Call the AI service with the conversation history
            response = ai.invoke(messages)
            # Record the end time once the response is received
            responseEndTime = time.monotonic()
            # Display the AI's response
            st.markdown(response.content)
            # Save the AI's response in session_state for persistent chat history
            st.session_state["messages"].append({"role": "ai", "content": response.content})
    
    # After receiving a response, calculate and display the response time.
    if responseEndTime:
        responseTime = responseEndTime - responseStartTime
        st.write(f"*(Last response took {':red[**' if responseTime > MAX_RESPONSE_TIME else ''}{responseTime:.4f} seconds{'**]' if responseTime > MAX_RESPONSE_TIME else ''})*")


# -----------------------------------------------------------------------------
# Function: main
# Purpose: Entry point for the application.
# -----------------------------------------------------------------------------
def main():
    # Reset the API key environment variable (could be modified as needed)
    os.environ["GROQ_API_KEY"] = ""
    # Render the main page of the app
    mainPage()


# Run the application
main()
