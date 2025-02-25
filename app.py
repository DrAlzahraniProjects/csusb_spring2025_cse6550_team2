import os
import time
import uuid
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from langchain_groq import ChatGroq

# Constants
COOLDOWN_CHECK_PERIOD = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN = 10
COOLDOWN_DURATION = 180.0
MAX_RESPONSE_TIME = 3.0
SHOWN_API_KEY_CHARACTERS_COUNT = 8

# Updated system prompt
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

# --- Callback Functions for Feedback Buttons ---

def update_like():
    gt = st.session_state.get("last_ground_truth", "yes")
    if gt == "yes":
        st.session_state["eval_data"]["y_true"].append("yes")
        st.session_state["eval_data"]["y_pred"].append("yes")
    else:
        st.session_state["eval_data"]["y_true"].append("unanswerable")
        st.session_state["eval_data"]["y_pred"].append("unanswerable")

def update_unlike():
    gt = st.session_state.get("last_ground_truth", "yes")
    if gt == "yes":
        st.session_state["eval_data"]["y_true"].append("yes")
        st.session_state["eval_data"]["y_pred"].append("no")
    else:
        st.session_state["eval_data"]["y_true"].append("unanswerable")
        st.session_state["eval_data"]["y_pred"].append("unanswerable")

def copy_response(text):
    """Embed a small HTML snippet to copy text to clipboard."""
    st.components.v1.html(f"""
    <script>
      navigator.clipboard.writeText({text!r});
    </script>
    """, height=0)

def speak_response(text):
    """Embed a small HTML snippet to speak text using the Web Speech API."""
    st.components.v1.html(f"""
    <script>
      var msg = new SpeechSynthesisUtterance({text!r});
      window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# --- Feedback Buttons (Native Streamlit) ---
def add_feedback_buttons(response_content: str):
    """
    Displays Copy, Like, Dislike, and Speech buttons in one row.
    The Like and Dislike buttons trigger evaluation callbacks.
    The Copy and Speech buttons trigger their own actions.
    Each button gets a unique key using uuid4.
    """
    unique_key = str(uuid.uuid4())
    col1, col2, col3, col4 = st.columns(4)
    col1.button("📋 Copy", key="copy_" + unique_key, on_click=copy_response, args=(response_content,))
    col2.button("👍 Like", key="like_" + unique_key, on_click=update_like)
    col3.button("👎 Dislike", key="dislike_" + unique_key, on_click=update_unlike)
    col4.button("🔊 Speech", key="speech_" + unique_key, on_click=speak_response, args=(response_content,))

def scroll_to_bottom():
    """Auto-scroll so the latest message is visible."""
    scroll_script = """
    <script>
    window.scrollTo(0, document.body.scrollHeight);
    </script>
    """
    st.components.v1.html(scroll_script, height=0)

def canAnswer() -> bool:
    currentTimestamp = time.monotonic()
    if st.session_state["cooldownBeginTimestamp"] is not None:
        if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
    else:
        st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:]
        st.session_state["messageTimes"].append(currentTimestamp)
        if (len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or 
            st.session_state["messageTimes"][-1] - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD):
            return True
        else:
            st.session_state["cooldownBeginTimestamp"] = currentTimestamp

    cooldownMinutes = int(COOLDOWN_CHECK_PERIOD // 60)
    cooldownSeconds = int(COOLDOWN_CHECK_PERIOD) % 60
    remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
    remainingMinutes = int(remainingTime // 60)
    remainingSeconds = int(remainingTime) % 60
    st.write(f"ERROR: You've reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} questions per " +
             f"{cooldownMinutes} minute{'s' if cooldownMinutes != 1 else ''} {cooldownSeconds} second{'s' if cooldownSeconds != 1 else ''}. " +
             f"Please try again in {remainingMinutes} minute{'s' if remainingMinutes != 1 else ''} " +
             f"{remainingSeconds} second{'s' if remainingSeconds != 1 else ''}.")
    return False

def apiBox():
    with st.container():
        if "GROQ_API_KEY" not in st.session_state or not st.session_state.get("GROQ_API_KEY"):
            st.write("**Groq API Key Setup**")
            api_key_placeholder = st.empty()
            newAPIkey = api_key_placeholder.text_input("New Groq API key:", placeholder="[New Groq API key]", type="password")
            if newAPIkey:
                st.session_state["GROQ_API_KEY"] = newAPIkey
                os.environ["GROQ_API_KEY"] = newAPIkey
                api_key_placeholder.empty()
                st.rerun()
        else:
            current_key = st.session_state["GROQ_API_KEY"]
            half_char_count = SHOWN_API_KEY_CHARACTERS_COUNT // 2
            masked_key = current_key if len(current_key) <= SHOWN_API_KEY_CHARACTERS_COUNT else (current_key[:half_char_count] + "..." + current_key[-(SHOWN_API_KEY_CHARACTERS_COUNT - half_char_count):])
            st.write(f"Current key: `{masked_key}`")
            if st.button("Clear API Key"):
                del st.session_state["GROQ_API_KEY"]
                os.environ.pop("GROQ_API_KEY", None)
                st.rerun()

def is_answerable(question: str) -> bool:
    """
    Returns True if the question contains at least one CSUSB-related keyword;
    otherwise returns False (i.e. the question is treated as unanswerable).
    """
    question_lower = question.lower()
    csusb_keywords = ["csusb", "california state university", "study abroad", "travel abroad", "exchange", "visa", "application", "scholarship", "living abroad"]
    if not any(kw in question_lower for kw in csusb_keywords):
        return False
    unanswerable_keywords = ["nursing", "fulbright", "concordia", "information session", "internal deadline"]
    for keyword in unanswerable_keywords:
        if keyword in question_lower:
            return False
    return True

def render_confusion_matrix_html() -> str:
    """
    Maps evaluation data to booleans: "yes" -> True; everything else ("no", "unanswerable") -> False.
    Then computes and returns an HTML string for the binary confusion matrix.
    """
    y_true = st.session_state["eval_data"]["y_true"]
    y_pred = st.session_state["eval_data"]["y_pred"]

    y_true_bool = [True if label=="yes" else False for label in y_true]
    y_pred_bool = [True if label=="yes" else False for label in y_pred]

    if len(y_true_bool) == 0:
        TP, FN, FP, TN = 0, 0, 0, 0
        accuracy = 0.0
        precision = 0.0
        sensitivity = 0.0
        f1 = 0.0
        specificity = 0.0
    else:
        cm = confusion_matrix(y_true_bool, y_pred_bool, labels=[True, False])
        TP, FN = cm[0, 0], cm[0, 1]
        FP, TN = cm[1, 0], cm[1, 1]
        accuracy = accuracy_score(y_true_bool, y_pred_bool)
        precision = precision_score(y_true_bool, y_pred_bool, pos_label=True, zero_division=0)
        sensitivity = recall_score(y_true_bool, y_pred_bool, pos_label=True, zero_division=0)
        f1 = f1_score(y_true_bool, y_pred_bool, pos_label=True, zero_division=0)
        cm_full = confusion_matrix(y_true_bool, y_pred_bool, labels=[False, True])
        TN_, FP_, FN_, TP_ = cm_full.ravel()
        specificity = TN_ / (TN_ + FP_) if (TN_ + FP_) else 0

    # Build the HTML with embedded CSS.
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        .confusion-container {{
          background-color: #f3cac3;
          color: #000;
          padding: 20px;
          border-radius: 8px;
          border: 1px solid #333;
          font-family: Arial, sans-serif;
          margin-bottom: 20px;
          width: 100%;
          box-sizing: border-box;
        }}
        .confusion-container h2 {{
          margin-top: 0;
          margin-bottom: 15px;
        }}
        .stats {{
          margin-bottom: 20px;
        }}
        .stats p {{
          margin: 5px 0;
        }}
        .table-container {{
          display: block;
          width: 100%;
          overflow-x: auto;
          margin-bottom: 20px;
          box-sizing: border-box;
        }}
        .confusion-table {{
          border: 2px solid #000;
          border-collapse: collapse;
          text-align: center;
          width: 100%;
        }}
        .confusion-table th,
        .confusion-table td {{
          border: 1px solid #000;
          padding: 5px;
        }}
        .confusion-table th {{
          background-color: #f8dcd7;
          white-space: normal;
          word-wrap: break-word;
          font-size: 0.9em;
        }}
        .reset-btn {{
          background-color: #fff;
          color: #000;
          padding: 10px 20px;
          border: 2px solid #000;
          cursor: pointer;
          transition: background-color 0.3s, color 0.3s;
        }}
        .reset-btn:hover {{
          background-color: #000;
          color: #fff;
        }}
      </style>
    </head>
    <body>
      <div class="confusion-container">
        <h2>Confusion Matrix (Accuracy: {accuracy:.2f})</h2>
        <div class="stats">
          <p><strong>Sensitivity (TP Rate):</strong> {sensitivity:.2f}</p>
          <p><strong>Specificity (TN Rate):</strong> {specificity:.2f}</p>
        </div>
        <div class="table-container">
          <table class="confusion-table">
            <tr>
              <th>Ground Truth</th>
              <th>Prediction (Detailed Answer)</th>
              <th>Prediction (Safe Disclaimer)</th>
            </tr>
            <tr>
              <th style="background-color: #f8dcd7;">Answerable (yes)</th>
              <td>{TP} (TP)</td>
              <td>{FN} (FN)</td>
            </tr>
            <tr>
              <th style="background-color: #f8dcd7;">Unanswerable</th>
              <td>{FP} (FP)</td>
              <td>{TN} (TN)</td>
            </tr>
          </table>
        </div>
        <div class="stats">
          <p><strong>Precision:</strong> {precision:.2f}</p>
          <p><strong>Recall:</strong> {sensitivity:.2f}</p>
          <p><strong>F1 Score:</strong> {f1:.2f}</p>
        </div>
        <button class="reset-btn" onclick="window.location.reload();">Reset</button>
      </div>
    </body>
    </html>
    """
    return html_code

def mainPage():
    st.markdown("""
        <style>
            body {
                background-color: #007BFF !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>", unsafe_allow_html=True)
    apiBox()

    if "cooldownBeginTimestamp" not in st.session_state:
        st.session_state["cooldownBeginTimestamp"] = None
    if "messageTimes" not in st.session_state:
        st.session_state["messageTimes"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "eval_data" not in st.session_state:
        st.session_state["eval_data"] = {"y_true": [], "y_pred": []}

    with st.sidebar:
        cm_placeholder = st.empty()
        cm_placeholder.markdown(render_confusion_matrix_html(), unsafe_allow_html=True)

    st.subheader("Chatbot")
    for msg in st.session_state["messages"]:
        display_role = "Alexy" if msg["role"] == "ai" else msg["role"]
        with st.chat_message(display_role):
            st.markdown(msg["content"])
            if msg["role"] == "ai":
                add_feedback_buttons(msg["content"])

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Please enter your Groq API key above to use the chatbot.")
        return

    ai = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )

    responseStartTime, responseEndTime = 0.0, 0.0
    prompt = st.chat_input("What is your question?")
    if prompt and canAnswer():
        st.chat_message("human").markdown(prompt)
        st.session_state["messages"].append({"role": "human", "content": prompt})
        # Ground truth: "yes" if CSUSB-related; otherwise, "unanswerable".
        ground_truth = "yes" if is_answerable(prompt) else "unanswerable"
        st.session_state["last_ground_truth"] = ground_truth

        messages = [("system", SYSTEM_PROMPT)] + [(m["role"], m["content"]) for m in st.session_state["messages"]]

        responseStartTime = time.monotonic()
        with st.chat_message("Alexy"):
            response = ai.invoke(messages)
            responseEndTime = time.monotonic()
            st.markdown(response.content)
            st.session_state["messages"].append({"role": "ai", "content": response.content})
            add_feedback_buttons(response.content)

        cm_placeholder.markdown(render_confusion_matrix_html(), unsafe_allow_html=True)
        scroll_to_bottom()

    if responseEndTime:
        responseTime = responseEndTime - responseStartTime
        time_label = f":red[**{responseTime:.4f} seconds**]" if responseTime > MAX_RESPONSE_TIME else f"{responseTime:.4f} seconds"
        st.write(f"*(Last response took {time_label})*")

def main():
    mainPage()

if __name__ == "__main__":
    main()
