from langchain_groq import ChatGroq
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import time  # Provides time-related functions
from typing import Literal, TypeAlias

AnswerTypes: TypeAlias = Literal["yes", "no", "unanswerable"]

# Constants for cooldown logic and response timing
COOLDOWN_CHECK_PERIOD = 60.0  # Time window in seconds to count messages for cooldown
MAX_MESSAGES_BEFORE_COOLDOWN = 10  # Maximum allowed messages within the check period before cooldown
COOLDOWN_DURATION = 180.0  # Duration (in seconds) of the cooldown period once the limit is reached
MAX_RESPONSE_TIME = 3.0  # Maximum acceptable response time in seconds before highlighting slow responses
SHOWN_API_KEY_CHARACTERS_COUNT = 8  # Number of characters from the API key to show (rest will be masked)
ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK = 30

# System prompt that instructs the ChatGroq model about its behavior and limitations
SYSTEM_PROMPT = """
You are Llama 3, an expert assistant for the Study Abroad program of California State University, San Bernardino (CSUSB).
Your purpose is to help students with all questions related to studying abroad. You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.
Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student. If a question is unrelated (e.g. politics or unrelated personal advice), politely guide the user back to study abroad topics.
- **No Negative Responses:** While you must remain truthful at all times, also avoid negative opinions, discouragement, or any response that may deter students from studying abroad.
- **Encourage and Inform:** Provide factual, detailed, and encouraging responses to all study abroad inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad, including politics, religion, or personal debates.
- You MUST begin every response with either the phrase "Yes", "No", or "I cannot answer that".
"""

ANSWERABLE_QUESTIONS: dict[str, AnswerTypes] = {
    "does csusb offer study abroad programs?": "yes",
    "can i apply for a study abroad program at csusb?": "yes",
    "is toronto a good place for students to live while studying abroad?": "yes",
    "do i need a visa to study at the university of seoul?": "yes",
    "can i study in south korea or taiwan if I only know english?": "yes"
}
CORRECT_ANSWER_KEYWORDS: tuple[str] = ("yes", "indeed", "correct", "right")
UNANSWERABLE_ANSWER_KEYWORDS: tuple[str] = ("i cannot answer", "i cannot help with", "i do not know")

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
                st.rerun()  # Rerun the script to show the setup section again

def render_confusion_matrix():
    """
    Renders a dark-themed confusion matrix table along with evaluation metrics.
    """
    y_true = st.session_state["eval_data"]["y_true"]
    y_pred = st.session_state["eval_data"]["y_pred"]

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    accuracy = accuracy_score(y_true, y_pred) if y_true and y_pred else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = TN / (TN + FP) if TN + FP else 0

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

    st.html(css + html_table)

def updateEvalData(question: str, givenAnswer: str) -> None:
    correctAnswerType = ANSWERABLE_QUESTIONS[question.strip().lower()].lower() if question.strip().lower() in ANSWERABLE_QUESTIONS else "unanswerable"
    if any(keyword.lower() in givenAnswer[:ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK].lower() for keyword in CORRECT_ANSWER_KEYWORDS): givenAnswerType = "yes"
    elif any(keyword.lower() in givenAnswer[:ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK].lower() for keyword in UNANSWERABLE_ANSWER_KEYWORDS): givenAnswerType = "unanswerable"
    else: givenAnswerType = "no"

    st.session_state["eval_data"]["y_true"].append(correctAnswerType != "unanswerable")
    st.session_state["eval_data"]["y_pred"].append(givenAnswerType != "unanswerable" and (correctAnswerType == "unanswerable" or givenAnswerType == correctAnswerType))


def mainPage():
    """Render the main page with the confusion matrix and chatbot."""
    
    st.html("""
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
    """)

    st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")

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
        st.html('<div class="center-chat">')
        st.subheader("Chatbot")

        # Display previous chat messages.
        for msg in st.session_state["messages"]:
            display_role = "ai" if msg["role"] == "ai" else msg["role"]
            with st.chat_message(display_role):
                st.markdown(msg["content"])

        # Retrieve the API key from the environment.
        api_key = st.session_state["GROQ_API_KEY"] if "GROQ_API_KEY" in st.session_state else None
        if not api_key:
            st.error("Please enter your Groq API key above to use the chatbot.")
            st.html("</div>")
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
            with st.chat_message("ai"):
                response = ai.invoke(messages)
                responseEndTime = time.monotonic()
                st.markdown(response.content)
                st.session_state["messages"].append({"role": "ai", "content": response.content})

            updateEvalData(prompt, response.content)
            with col_left: render_confusion_matrix()
            scroll_to_bottom()

        if responseEndTime:
            responseTime = responseEndTime - responseStartTime
            time_label = (
                f":red[**{responseTime:.4f} seconds**]" 
                if responseTime > MAX_RESPONSE_TIME 
                else f"{responseTime:.4f} seconds"
            )
            st.write(f"*(Last response took {time_label})*")

        st.html("</div>")

def main():
    """Entry point for the Streamlit app."""
    mainPage()

if __name__ == "__main__":
    main()
