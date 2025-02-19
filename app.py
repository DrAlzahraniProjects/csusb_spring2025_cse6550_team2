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

# Updated system prompt with guidelines on answerable and unanswerable questions.
SYSTEM_PROMPT = """
You are an expert assistant for the Study Abroad program of California State University, San Bernardino (CSUSB).
Your name is Alexy.
You are designed to help students with all questions related to studying abroad.
 - Provide a concise and accurate answer based solely on the context above.
        - If the context does not contain enough information to answer the question, respond with:
        "I don’t have enough information to answer this question."
        - Do not generate, assume, or make up any details beyond the given context.
You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.

Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student.
- **Answering Guidelines:**
    * If a student's question falls under one of these approved topics, provide a detailed and correct answer:
        1. **What study abroad programs does CSUSB offer?**  
           Explain CSUSB-approved programs, including exchange partnerships and direct enrollment options.
        2. **How do I apply for a study abroad program?**  
           Outline the step-by-step application process, including deadlines and required documents.
        3. **How is the living situation in Toronto?**  
           Provide details about living in Toronto such as popular student neighborhoods, housing options, cost of living, and budgeting tips.
        4. **What's the visa process if I get admission to the University of Seoul?**  
           Explain the visa application process for students admitted to the University of Seoul, including visa types, required documents, and application steps.
        5. **If I know only one language that is English, can I study in South Korea or Taiwan?**  
           Provide information about English-taught programs, language support services, and cultural adaptation programs available in South Korea and Taiwan.
    * If a student's question falls into any of these topics that you are not allowed to answer:
        - When will the Study Abroad 101 information sessions be held?
        - When is the deadline to apply for Concordia University for the summer semester?
        - Which universities are under CSUSB-approved direct enrollment programs?
        - Which scholarships are available under the study abroad program?
        - What is the internal deadline for the Fulbright Scholarship application set by CSUSB?
      Then, respond with a message stating that you cannot provide that information and advise the student to refer to the official CSUSB Study Abroad website or contact the Study Abroad Office.
- **No Negative Responses:** Remain factual and avoid any discouraging language.
- **Encourage and Inform:** Provide clear, supportive, and correct responses to the approved inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad (e.g., politics, religion, or personal debates).
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
    """Auto-scroll so the latest message is visible."""
    scroll_script = """
    <script>
    window.scrollTo(0, document.body.scrollHeight);
    </script>
    """
    components.html(scroll_script, height=0)

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
            api_key_placeholder = st.empty()  # Create a placeholder for the input box
            newAPIkey = api_key_placeholder.text_input(
                "New Groq API key:",
                placeholder="[New Groq API key]",
                type="password",
            )
            # If the user enters a new API Key, update session state and environment variables
            if newAPIkey:
                st.session_state["GROQ_API_KEY"] = newAPIkey
                os.environ["GROQ_API_KEY"] = newAPIkey
                api_key_placeholder.empty()  # Clear the placeholder to hide the input box
                st.rerun()  # Rerun the script to update the UI
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

def is_answerable(question: str) -> bool:
    """
    Determines if the question is answerable by the chatbot.
    Returns True if the question is about study abroad topics (i.e. expects a detailed answer),
    and False if it pertains to topics that should be answered with a safe disclaimer.
    """
    question_lower = question.lower()
    unanswerable_keywords = [
        "nursing", "fulbright", "concordia", "information session", "internal deadline"
    ]
    for keyword in unanswerable_keywords:
        if keyword in question_lower:
            return False
    return True

def evaluate_response_context(response: str, question: str) -> bool:
    """
    Evaluates whether the generated response is contextually appropriate.
    
    For answerable questions (is_answerable(question) == True):
      - Returns True if the response contains at least two context-specific keywords.
    For unanswerable questions (is_answerable(question) == False):
      - Returns False if the response contains a safe disclaimer such as 
        "I don’t have enough information" or "please refer" (which is the expected response).
    """
    response_lower = response.lower()
    
    if not is_answerable(question):
        # For unanswerable questions, a safe disclaimer is expected.
        if "i don’t have enough information" in response_lower or "please refer" in response_lower:
            return False
        else:
            return True
    else:
        # For answerable questions, check for relevant context-specific keywords.
        # Expanded list of keywords to capture diverse valid responses.
        context_keywords = [
            "csusb", "study abroad", "application", "visa", "housing", "exchange",
            "english", "south korea", "university", "program", "language", "international"
        ]
        matches = sum(1 for kw in context_keywords if kw in response_lower)
        return True if matches >= 2 else False

###############################################################################
# Confusion matrix function that returns HTML as a string.
###############################################################################
def render_confusion_matrix_html() -> str:
    """
    Generates the confusion matrix HTML code as a string.
    
    Ground truth:
      - True: Answerable question (detailed answer expected).
      - False: Unanswerable question (safe disclaimer expected).
    """
    y_true = st.session_state["eval_data"]["y_true"]
    y_pred = st.session_state["eval_data"]["y_pred"]

    if len(y_true) == 0:
        return "<p>No evaluation data yet.</p>"

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    TP = cm[0, 0]  # Actual True, Predicted True
    FN = cm[0, 1]  # Actual True, Predicted False
    FP = cm[1, 0]  # Actual False, Predicted True
    TN = cm[1, 1]  # Actual False, Predicted False

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)

    cm_full = confusion_matrix(y_true, y_pred, labels=[False, True])
    TN_, FP_, FN_, TP_ = cm_full.ravel()
    specificity = TN_ / (TN_ + FP_) if (TN_ + FP_) else 0

    html_code = f"""
    <div style="background-color: #f3cac3; color: #000; padding: 20px; border-radius: 6px; border: 1px solid #333; margin-bottom: 20px;">
      <h2 style="margin-top: 0; margin-bottom: 15px;">Confusion Matrix</h2>
      <div style="margin-bottom: 20px;">
        <p><strong>Sensitivity (True Positive Rate):</strong> {sensitivity:.2f}</p>
        <p><strong>Specificity (True Negative Rate):</strong> {specificity:.2f}</p>
      </div>
      <div style="overflow-x: auto; width: 100%;">
        <table style="width: 100%; border: 2px solid #000; border-collapse: collapse; text-align: center; margin-bottom: 20px;">
          <tr style="background-color: #f8dcd7;">
            <th style="border: 1px solid #000;"></th>
            <th style="border: 1px solid #000;">Predicted True (Detailed Answer)</th>
            <th style="border: 1px solid #000;">Predicted False (Safe Disclaimer)</th>
          </tr>
          <tr>
            <th style="border: 1px solid #000; background-color: #f8dcd7;">Actual True<br>(Answerable Question)</th>
            <td style="border: 1px solid #000;">{TP} (TP)</td>
            <td style="border: 1px solid #000;">{FN} (FN)</td>
          </tr>
          <tr>
            <th style="border: 1px solid #000; background-color: #f8dcd7;">Actual False<br>(Unanswerable Question)</th>
            <td style="border: 1px solid #000;">{FP} (FP)</td>
            <td style="border: 1px solid #000;">{TN} (TN)</td>
          </tr>
        </table>
      </div>
      <div style="margin-bottom: 20px;">
        <p><strong>Accuracy:</strong> {accuracy:.2f}</p>
        <p><strong>Precision:</strong> {precision:.2f}</p>
        <p><strong>Recall (Sensitivity):</strong> {sensitivity:.2f}</p>
        <p><strong>F1 Score:</strong> {f1:.2f}</p>
      </div>
      <button 
        style="background-color: #fff; color: #000; padding: 10px 20px; border: 2px solid #000; cursor: pointer;"
        onclick="window.location.reload();">
        Reset
      </button>
    </div>
    """
    return html_code

###############################################################################
# Main page function
###############################################################################
def mainPage():
    """Render the main page with the confusion matrix and chatbot."""
    
    st.html("""
        <style>
            body {
                background-color: #007BFF !important;
                color: white !important;
            }
        </style>
    """)

    st.markdown("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>", unsafe_allow_html=True)
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

    # Place the confusion matrix in the sidebar so it's always visible.
    with st.sidebar:
        cm_placeholder = st.empty()
        cm_placeholder.markdown(render_confusion_matrix_html(), unsafe_allow_html=True)

    # Main chat area.
    st.subheader("Chatbot")
    for msg in st.session_state["messages"]:
        display_role = "Alexy" if msg["role"] == "ai" else msg["role"]
        with st.chat_message(display_role):
            st.markdown(msg["content"])

    # Retrieve API key.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Please enter your Groq API key above to use the chatbot.")
        return

    # Instantiate the ChatGroq model.
    ai = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )

    responseStartTime, responseEndTime = 0.0, 0.0

    # Chat input.
    prompt = st.chat_input("What is your question?")
    if prompt and canAnswer():
        st.chat_message("human").markdown(prompt)
        st.session_state["messages"].append({"role": "human", "content": prompt})

        # Determine ground truth.
        ground_truth = is_answerable(prompt)

        messages = [("system", SYSTEM_PROMPT)] + [(m["role"], m["content"]) for m in st.session_state["messages"]]

        responseStartTime = time.monotonic()
        with st.chat_message("Alexy"):
            response = ai.invoke(messages)
            responseEndTime = time.monotonic()
            st.markdown(response.content)
            st.session_state["messages"].append({"role": "ai", "content": response.content})

        # Evaluate the response.
        predicted = evaluate_response_context(response.content, prompt)

        st.session_state["eval_data"]["y_true"].append(ground_truth)
        st.session_state["eval_data"]["y_pred"].append(predicted)

        # Update the confusion matrix in the sidebar.
        cm_placeholder.markdown(render_confusion_matrix_html(), unsafe_allow_html=True)

        scroll_to_bottom()

    if responseEndTime:
        responseTime = responseEndTime - responseStartTime
        time_label = (
            f":red[**{responseTime:.4f} seconds**]" 
            if responseTime > MAX_RESPONSE_TIME 
            else f"{responseTime:.4f} seconds"
        )
        st.write(f"*(Last response took {time_label})*")

def main():
    """Entry point for the Streamlit app."""
    mainPage()

if __name__ == "__main__":
    main()
