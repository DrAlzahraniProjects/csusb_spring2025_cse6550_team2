import os  # Provides functions to interact with the operating system
import time  # Provides time-related functions
import random  # Provides functions for generating random numbers
import streamlit as st  
import streamlit.components.v1 as components  # Streamlit library for building web apps

# Scikit-learn metrics for evaluating model performance (e.g., confusion matrix)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# For interacting with the Groq API (used to access the chat AI model)
from langchain_groq import ChatGroq

# Constants for cooldown logic and response timing
COOLDOWN_CHECK_PERIOD = 60.0  # Time window in seconds to count messages for cooldown
MAX_MESSAGES_BEFORE_COOLDOWN = 10  # Maximum allowed messages within the check period before cooldown
COOLDOWN_DURATION = 180.0  # Duration (in seconds) of the cooldown period once the limit is reached
MAX_RESPONSE_TIME = 3.0  # Maximum acceptable response time in seconds before highlighting slow responses
SHOWN_API_KEY_CHARACTERS_COUNT = 8  # Number of characters from the API key to show (rest will be masked)

# System prompt that instructs the ChatGroq model about its behavior and limitations
SYSTEM_PROMPT = """
You are an expert assistant for the Study Abroad program of California State University, San Bernardino (CSUSB).
Your name is Alexy.
You are designed to help students with all questions related to studying abroad.
You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.
Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student. If a question is unrelated (e.g. politics or unrelated personal advice), politely guide the user back to study abroad topics.
- **No Negative Responses:** While you must remain truthful at all times, also avoid negative opinions, discouragement, or any response that may deter students from studying abroad.
- **Encourage and Inform:** Provide factual, detailed, and encouraging responses to all study abroad inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad, including politics, religion, or personal debates.
"""

def scroll_to_bottom():
    # Auto-scroll so the latest message is visible
    if 'scroll' not in st.session_state:
        st.session_state['scroll'] = 0
    st.session_state['scroll'] += 1
    st.rerun()

def canAnswer() -> bool:
    """Check if user can send a new message based on cooldown logic."""
    currentTimestamp = time.monotonic()  # Get the current time in seconds
    if st.session_state["cooldownBeginTimestamp"] is not None:
        if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
    else:
        st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:]
        st.session_state["messageTimes"].append(currentTimestamp)
        if (
            len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN
            or st.session_state["messageTimes"][-1]
            - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1]
            >= COOLDOWN_CHECK_PERIOD
        ):
            return True
        else:
            st.session_state["cooldownBeginTimestamp"] = currentTimestamp

    cooldownMinutes = int(COOLDOWN_CHECK_PERIOD // 60)
    cooldownSeconds = int(COOLDOWN_CHECK_PERIOD) % 60
    remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
    remainingMinutes = int(remainingTime // 60)
    remainingSeconds = int(remainingTime) % 60
    st.write(
        f"ERROR: You've reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} questions per "
        f"{cooldownMinutes} minute{'s' if cooldownMinutes != 1 else ''} {cooldownSeconds} second{'s' if cooldownSeconds != 1 else ''}. "
        f"Please try again in {remainingMinutes} minute{'s' if remainingMinutes != 1 else ''} "
        f"{remainingSeconds} second{'s' if remainingSeconds != 1 else ''}."
    )
    return False

# def apiBox():
#     """Display and update the API key input box for the Groq API."""
#     with st.container():
#         st.write("**Groq API Key Setup**")
#         newAPIkey = st.text_input(
#             "New Groq API key:",
#             placeholder="[New Groq API key]",
#             type="password",
#         )
#         if newAPIkey:
#             st.session_state["GROQ_API_KEY"] = newAPIkey
#             os.environ["GROQ_API_KEY"] = newAPIkey

#         if "GROQ_API_KEY" in os.environ and os.environ["GROQ_API_KEY"]:
#             current_key = os.environ["GROQ_API_KEY"]
#             half_char_count = SHOWN_API_KEY_CHARACTERS_COUNT // 2
#             if len(current_key) <= SHOWN_API_KEY_CHARACTERS_COUNT:
#                 masked_key = current_key
#             else:
#                 masked_key = (
#                     current_key[:half_char_count] + "..." +
#                     current_key[-(SHOWN_API_KEY_CHARACTERS_COUNT - half_char_count):]
#                 )
#             st.write(f"Current key: `{masked_key}`")
#         else:
#             st.write("Current key: [none provided]")


def apiBox():
    """Display and update the API key input box for the Groq API."""
    with st.container():
        # Only show the setup section if there is no API Key in session state
        if "GROQ_API_KEY" not in st.session_state or not st.session_state.get("GROQ_API_KEY"):
            st.write("**Groq API Key Setup**")  # Show the setup title
            newAPIkey = st.text_input(
                "New Groq API key:",
                placeholder="[New Groq API key]",
                type="password",
            )
            # If the user enters a new API Key, update session state and environment variables
            if newAPIkey:
                st.session_state["GROQ_API_KEY"] = newAPIkey
                os.environ["GROQ_API_KEY"] = newAPIkey
                st.rerun()  # Rerun the script to hide the input box and setup title
        else:
            # If an API Key exists, display the current key (partially masked)
            current_key = st.session_state["GROQ_API_KEY"]
            half_char_count = SHOWN_API_KEY_CHARACTERS_COUNT // 2
            if len(current_key) <= SHOWN_API_KEY_CHARACTERS_COUNT:
                masked_key = current_key
            else:
                masked_key = (
                    current_key[:half_char_count] + "..." +
                    current_key[-(SHOWN_API_KEY_CHARACTERS_COUNT - half_char_count):]
                )
            st.write(f"Current key: `{masked_key}`")

            # Provide a button to clear the current API Key
            if st.button("Clear API Key"):
                del st.session_state["GROQ_API_KEY"]
                os.environ.pop("GROQ_API_KEY", None)
                st.rerun()  # Rerun the script to show the setup section again

def render_confusion_matrix():
    """
    Renders a dark-themed confusion matrix table along with evaluation metrics.
    """
    y_true = st.session_state["eval_data"]["y_true"]
    y_pred = st.session_state["eval_data"]["y_pred"]

    if len(y_true) == 0:
        st.write("No evaluation data yet.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN_, FP_, FN_, TP_ = cm_full.ravel()
    specificity = TN_ / (TN_ + FP_) if (TN_ + FP_) else 0

    css = """
    <style>
    .cm-container {
        background-color: #2F2F2F;
        border-radius: 8px;
        padding: 30px;
        color: #FFF;
        margin-top: 20px;
        border: 3px solid #FFF;
    }
    .cm-title {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 10px;
        text-align: left;
    }
    .cm-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }
    .cm-table th, .cm-table td {
        border: 2px solid #555;
        padding: 12px;
        text-align: center;
        font-size: 1rem;
    }
    .cm-table th {
        background-color: #3F3F3F;
    }
    .cm-metrics {
        margin-top: 25px;
        font-size: 1.1rem;
    }
    .metric-box {
        background-color: #3F3F3F;
        border-radius: 6px;
        padding: 10px 15px;
        margin: 10px 0;
        border: 1px solid #666;
    }
    </style>
    """

    html_table = f"""
    <div class="cm-container">
      <div class="cm-title">Confusion Matrix</div>
      <table class="cm-table">
        <tr>
          <th></th>
          <th colspan="2">Predicted</th>
        </tr>
        <tr>
          <th></th>
          <th>+</th>
          <th>-</th>
        </tr>
        <tr>
          <th>Actual +</th>
          <td>{TP} (TP)</td>
          <td>{FN} (FN)</td>
        </tr>
        <tr>
          <th>Actual -</th>
          <td>{FP} (FP)</td>
          <td>{TN} (TN)</td>
        </tr>
      </table>
      <div class="cm-metrics">
        <div class="metric-box">
          <strong>Sensitivity (true positive rate):</strong> {sensitivity:.2f}
        </div>
        <div class="metric-box">
          <strong>Specificity (true negative rate):</strong> {specificity:.2f}
        </div>
        <div class="metric-box">
          <strong>Accuracy:</strong> {accuracy:.2f}
        </div>
        <div class="metric-box">
          <strong>Precision:</strong> {precision:.2f}
        </div>
        <div class="metric-box">
          <strong>F1 Score:</strong> {f1:.2f}
        </div>
      </div>
    </div>
    """

    st.markdown(css + html_table, unsafe_allow_html=True)

def mainPage():
    """Render the main page with the confusion matrix and chatbot."""
    
    st.markdown("""
        <style>
            body {
                background-color: #007BFF !important;
                color: white !important;
            }
            .center-chat {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>", unsafe_allow_html=True)

    # Display the API key input box.
    apiBox()

    # Initialize session state variables if needed.
    if "cooldownBeginTimestamp" not in st.session_state:
        st.session_state["cooldownBeginTimestamp"] = None
    if "messageTimes" not in st.session_state:
        st.session_state["messageTimes"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "eval_data" not in st.session_state:
        st.session_state["eval_data"] = {"y_true": [], "y_pred": []}

    # Create two columns: one for the confusion matrix and one for the chat.
    col_left, col_center = st.columns([2, 3])

    with col_left:
        render_confusion_matrix()

    with col_center:
        st.markdown('<div class="center-chat">', unsafe_allow_html=True)
        st.subheader("Chatbot")

        # Display previous chat messages.
        for msg in st.session_state["messages"]:
            display_role = "Alexy" if msg["role"] == "ai" else msg["role"]
            with st.chat_message(display_role):
                st.markdown(msg["content"])

        # Retrieve the API key from the environment.
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("Please enter your Groq API key above to use the chatbot.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Instantiate the ChatGroq model using the provided API key.
        ai = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key,  # Pass the API key explicitly.
        )

        responseStartTime, responseEndTime = 0.0, 0.0

        # Chat input for the user.
        prompt = st.chat_input("What is your question?")
        if prompt and canAnswer():
            st.chat_message("human").markdown(prompt)
            st.session_state["messages"].append({"role": "human", "content": prompt})

            messages = [("system", SYSTEM_PROMPT)] + [(m["role"], m["content"]) for m in st.session_state["messages"]]

            responseStartTime = time.monotonic()
            with st.chat_message("Alexy"):
                response = ai.invoke(messages)
                responseEndTime = time.monotonic()
                st.markdown(response.content)
                st.session_state["messages"].append({"role": "ai", "content": response.content})

            # Simulate evaluation data.
            ground_truth = random.choice([1, 0])
            outcome = random.choice(["correct", "incorrect"])
            if ground_truth == 1:
                predicted = 1 if outcome == "correct" else 0
            else:
                predicted = 0 if outcome == "correct" else 1

            st.session_state["eval_data"]["y_true"].append(ground_truth)
            st.session_state["eval_data"]["y_pred"].append(predicted)

            scroll_to_bottom()

        if responseEndTime:
            responseTime = responseEndTime - responseStartTime
            time_label = (
                f":red[**{responseTime:.4f} seconds**]" 
                if responseTime > MAX_RESPONSE_TIME 
                else f"{responseTime:.4f} seconds"
            )
            st.write(f"*(Last response took {time_label})*")

        st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Entry point for the Streamlit app."""
    mainPage()

if __name__ == "__main__":
    main()
