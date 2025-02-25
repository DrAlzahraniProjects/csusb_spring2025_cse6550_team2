import os
# Ensure the "data" folder exists for storing scraped output and FAISS index.
os.makedirs("data", exist_ok=True)

import time
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util  # For evaluation purposes

# Import the retrieval chain function from your langchain integration file.
from scripts.langchain_integration import run_qa_chain
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Initialize an embedding model for evaluation purposes.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants
COOLDOWN_CHECK_PERIOD = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN = 10
COOLDOWN_DURATION = 180.0
MAX_RESPONSE_TIME = 3.0
SHOWN_API_KEY_CHARACTERS_COUNT = 8

# Updated system prompt instructing domain-specific responses.
SYSTEM_PROMPT = """
You are an AI expert assistant specialized in study abroad programs for California State University, San Bernardino (CSUSB). Your responses should be based solely on the information provided on https://goabroad.csusb.edu/ and any related context. 
- Only answer questions regarding CSUSB’s study abroad programs.
- If a question does not pertain to CSUSB’s study abroad programs, respond with "I don’t have enough information to answer this question."
- Do not generate or assume details beyond the verified content from https://goabroad.csusb.edu/.
"""
def format_response(output: dict) -> str:
   # query = output.get('query', 'No query provided')
    result = output.get('result', 'No result provided')
    lines = result.split('\n')
    #points = [line.strip() for line in lines if line.strip() and line.strip()[0].isdigit() and line.strip()[1:3] == ". "] 
    result_formatted = result.replace("\n", "<br>")
    formatted = (
        "<div style='text-align: left; margin-left: 2em;'>"
        "<h3 style='margin-bottom: 0.5em;'>Answer:</h3>"
        f"<p style='margin-left: 1em;'>{result_formatted}</p>"
        "</div>"
    )
    return formatted        
    #formatted = (
    #   "<div style='text-align: left; margin-left: 2em;'>"
    #    f"<h3 style='margin-bottom: 0.5em;'>Answer:</h3>"
    #    f"<p style='margin-left: 1em;'>{result}</p>"
    #    "</div>"
    #)
    #return formatted


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
    currentTimestamp = time.monotonic()
    if st.session_state.get("cooldownBeginTimestamp") is not None:
        if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
    else:
        st.session_state["messageTimes"] = st.session_state.get("messageTimes", [])
        st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:]
        st.session_state["messageTimes"].append(currentTimestamp)
        if (
            len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or
            st.session_state["messageTimes"][-1] - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD
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
        if "GROQ_API_KEY" not in st.session_state or not st.session_state.get("GROQ_API_KEY"):
            st.write("**Groq API Key Setup**")
            api_key_placeholder = st.empty()
            newAPIkey = api_key_placeholder.text_input(
                "New Groq API key:",
                placeholder="[New Groq API key]",
                type="password",
            )
            if newAPIkey:
                st.session_state["GROQ_API_KEY"] = newAPIkey
                os.environ["GROQ_API_KEY"] = newAPIkey
                api_key_placeholder.empty()
                st.rerun()
        else:
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
            if st.button("Clear API Key"):
                del st.session_state["GROQ_API_KEY"]
                os.environ.pop("GROQ_API_KEY", None)
                st.experimental_rerun()

def is_answerable(question: str) -> bool:
    """Determines if the question is answerable by the chatbot."""
    question_lower = question.lower()
    unanswerable_keywords = [
        "nursing", "fulbright", "concordia", "information session", "internal deadline"
    ]
    for keyword in unanswerable_keywords:
        if keyword in question_lower:
            return False
    return True

def evaluate_response_context(response: str, question: str) -> bool:
    """Evaluates whether the generated response is contextually appropriate."""
    #response_lower = response.lower()
    if isinstance(response, dict):
        response_str = response.get("result", str(response))
    else:
        response_str = str(response)
    
    response_lower = response_str.lower()
    
    if not is_answerable(question):
        if "i don’t have enough information" in response_lower or "please refer" in response_lower:
            return False
        return True
    else:
        context_keywords = ["csusb", "study abroad", "application", "visa", "housing", "exchange",
                            "english", "south korea", "university", "program", "language", "international"]
        matches = sum(1 for kw in context_keywords if kw in response_lower)
        return matches >= 2

def evaluate_retrieval(retrieved_text: str, actual_text: str, threshold: float = 0.8) -> bool:
    """Compute cosine similarity between retrieved and actual text."""
    embedding1 = embedding_model.encode(retrieved_text, convert_to_tensor=True)
    embedding2 = embedding_model.encode(actual_text, convert_to_tensor=True)
    cosine_sim = util.cos_sim(embedding1, embedding2).item()
    return cosine_sim >= threshold

def get_ground_truth(query: str) -> str:
    """Return a ground truth response for specific queries (for evaluation)."""
    ground_truth_dict = {
        "what study abroad programs does csusb offer?":
            "CSUSB offers various study abroad programs including short-term exchanges, semester-long programs, and faculty-led initiatives coordinated by the Office of International Programs.",
        # Add more pairs as needed.
    }
    return ground_truth_dict.get(query.lower(), "")

def render_confusion_matrix_html() -> str:
    """Generates HTML for the confusion matrix."""
    y_true = st.session_state["eval_data"]["y_true"]
    y_pred = st.session_state["eval_data"]["y_pred"]

    if len(y_true) == 0:
        return "<p>No evaluation data yet.</p>"

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)

    cm_full = confusion_matrix(y_true, y_pred, labels=[False, True])
    TN_, FP_, FN_, TP_ = cm_full.ravel()
    specificity = TN_ / (TN_ + FP_) if (TN_ + FP_) else 0

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
        <h2>Confusion Matrix</h2>
        <div class="stats">
          <p><strong>Sensitivity (True Positive Rate):</strong> {sensitivity:.2f}</p>
          <p><strong>Specificity (True Negative Rate):</strong> {specificity:.2f}</p>
        </div>
        <div class="table-container">
          <table class="confusion-table">
            <tr>
              <th></th>
              <th>Predicted True<br>(Detailed Answer)</th>
              <th>Predicted False<br>(Safe Disclaimer)</th>
            </tr>
            <tr>
              <th style="background-color: #f8dcd7;">Actual True<br>(Answerable Question)</th>
              <td>{TP} (TP)</td>
              <td>{FN} (FN)</td>
            </tr>
            <tr>
              <th style="background-color: #f8dcd7;">Actual False<br>(Unanswerable Question)</th>
              <td>{FP} (FP)</td>
              <td>{TN} (TN)</td>
            </tr>
          </table>
        </div>
        <div class="stats">
          <p><strong>Accuracy:</strong> {accuracy:.2f}</p>
          <p><strong>Precision:</strong> {precision:.2f}</p>
          <p><strong>Recall (Sensitivity):</strong> {sensitivity:.2f}</p>
          <p><strong>F1 Score:</strong> {f1:.2f}</p>
        </div>
        <button class="reset-btn" onclick="window.location.reload();">Reset</button>
      </div>
    </body>
    </html>
    """
    return html_code

def add_feedback_buttons(response_content: str):
    """Adds copy, like, dislike, and speech buttons below the response."""
    feedback_script = f"""
    <script>
    function copyToClipboard(text, button) {{
        navigator.clipboard.writeText(text).then(function() {{
            button.innerHTML = '<i class="fas fa-check"></i>';
            setTimeout(function() {{
                button.innerHTML = '<i class="fas fa-copy"></i>';
            }}, 1500);
        }}, function(err) {{
            console.error('Failed to copy text: ', err);
        }});
    }}

    function handleFeedback(button, type) {{
        const likeButton = document.getElementById('like-button');
        const dislikeButton = document.getElementById('dislike-button');
        if (button.style.color === 'red') {{
            button.style.color = 'white';
        }} else {{
            button.style.color = 'red';
            if (type === 'like') {{
                dislikeButton.style.color = 'white';
            }} else {{
                likeButton.style.color = 'white';
            }}
        }}
    }}

    function toggleSpeech(button, text) {{
        if (button.style.color === 'red') {{
            window.speechSynthesis.cancel();
            button.style.color = 'white';
            button.innerHTML = '<i class="fas fa-volume-up"></i>';
        }} else {{
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
            button.style.color = 'red';
            button.innerHTML = '<i class="fas fa-volume-off"></i>';
        }}
    }}
    </script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <div style="display: flex; gap: 8px; margin-top: -6px;">
        <button 
            style="background-color: gray; color: white; border: none; padding: 8px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center;" 
            onclick="copyToClipboard(`{response_content}`, this)">
            <i class="fas fa-copy"></i>
        </button>
        <button 
            id="like-button"
            style="background-color: gray; color: white; border: none; padding: 8px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center;" 
            onclick="handleFeedback(this, 'like')">
            <i class="fas fa-thumbs-up"></i>
        </button>
        <button 
            id="dislike-button"
            style="background-color: gray; color: white; border: none; padding: 8px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center;" 
            onclick="handleFeedback(this, 'dislike')">
            <i class="fas fa-thumbs-down"></i>
        </button>
        <button 
            id="speech-button"
            style="background-color: gray; color: white; border: none; padding: 8px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center;" 
            onclick="toggleSpeech(this, `{response_content}`)">
            <i class="fas fa-volume-up"></i>
        </button>
    </div>
    """
    components.html(feedback_script, height=40)

def mainPage():
    """Render the main page with the confusion matrix and chatbot."""
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
        display_role = "ai" if msg["role"] == "ai" else msg["role"]
        with st.chat_message(display_role):
            st.markdown(msg["content"])
            if msg["role"] == "ai":
                add_feedback_buttons(msg["content"])

    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Please enter your Groq API key above to use the chatbot.")
        return

    responseStartTime, responseEndTime = 0.0, 0.0

    prompt = st.chat_input("What is your question?")
    if prompt and canAnswer():
        st.chat_message("human").markdown(prompt)
        st.session_state["messages"].append({"role": "human", "content": prompt})

        # Create the message list including the system prompt.
        messages = [("system", SYSTEM_PROMPT)] + [(m["role"], m["content"]) for m in st.session_state["messages"]]

        responseStartTime = time.monotonic()
        with st.chat_message("ai"):
            # Instead of directly invoking the ChatGroq API here,
            # we call our retrieval chain function to get the domain-specific answer.
            try:
                # run_qa_chain() is imported from langchain_integration.
                response = run_qa_chain(prompt)
                # If the response is a dict, format it nicely
                if isinstance(response, dict):
                    formatted_response = format_response(response)
                else:
                    formatted_response = response
            except Exception as e:
                st.error(f"Error retrieving response: {e}")
                formatted_response = "Sorry, an error occurred while fetching the answer."
            st.markdown(formatted_response, unsafe_allow_html=True)
            st.session_state["messages"].append({"role": "ai", "content": formatted_response})
            responseEndTime = time.monotonic()
            st.markdown(response)
            st.session_state["messages"].append({"role": "ai", "content": response})
            add_feedback_buttons(response)

        # Evaluation: using ground truth if available; otherwise, fallback.
        actual_answer = get_ground_truth(prompt)
        if actual_answer:
            predicted = evaluate_retrieval(response, actual_answer, threshold=0.8)
            st.session_state["eval_data"]["y_true"].append(True)
            st.session_state["eval_data"]["y_pred"].append(predicted)
        else:
            predicted = evaluate_response_context(response, prompt)
            st.session_state["eval_data"]["y_true"].append(is_answerable(prompt))
            st.session_state["eval_data"]["y_pred"].append(predicted)

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
