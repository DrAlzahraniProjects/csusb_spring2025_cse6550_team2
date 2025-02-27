from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import streamlit.components.v1 as components
import time

# Constants
COOLDOWN_CHECK_PERIOD = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN = 10
COOLDOWN_DURATION = 180.0
MAX_RESPONSE_TIME = 3.0
ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK = 30
MAX_QUESTIONS_TO_ASK: tuple[int | None, int | None] = (1, 1)
DEBUG_MODE: bool = False

# Updated system prompt
SYSTEM_PROMPT = """
You are Beta, an expert assistant for the Study Abroad program of California State University, San Bernardino (CSUSB).
You are designed to help students with all questions related to studying abroad.
You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.

Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student.
- **No Negative Responses:** Remain factual and avoid discouraging language.
- **Encourage and Inform:** Provide clear, supportive, and correct responses to the approved inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad (e.g., politics, religion, or personal debates).
- You MUST begin every response with either the phrase "Yes", "No", or "I don't have enough information to answer this question".

Provide a concise and accurate answer based solely on the context below.
If the context does not contain enough information to answer the question, respond with "I don't have enough information to answer this question." Do not generate, assume, or make up any details beyond the given context.

"""

ANSWERABLE_QUESTIONS: tuple[str, ...] = (
    "Does CSUSB offer Study Abroad programs?",
    "Can I apply for a Study Abroad program at CSUSB?",
    "Is Toronto a good place for students to live while studying abroad?",
    "Do I need a visa to study at the University of Seoul?",
    "Can I study in South Korea or Taiwan if I only know English?"
)
UNANSWERABLE_QUESTIONS: tuple[str, ...] = (
    "Is there a set date for the Study Abroad 101 information sessions?",
    "Is the application deadline for Concordia University's summer semester available here?",
    "Does the chatbot provide a full list of CSUSB-approved direct enrollment universities?",
    "Does the chatbot list all available study abroad scholarships?",
    "Is the internal deadline for the Fulbright Scholarship application set by CSUSB available here?"
)
CORRECT_ANSWER_KEYWORDS: tuple[str, ...] = ("yes", "indeed", "correct", "right")
UNANSWERABLE_ANSWER_KEYWORDS: tuple[str, ...] = ("cannot answer", "can't answer", "cannot help with", "cannot help you with", "can't help with", "can't help you with", "do not know", "don't know", "do not have enough info", "don't have enough info", "not knowledgable", "please refer", "don't have access", "do not have access", "cannot access", "can't access")
# Initialize an embedding model for evaluation purposes.
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
INDEX_PATH: str | None = os.path.join("data", "index")

# def format_response(output: dict) -> str:
#    # query = output.get('query', 'No query provided')
#     result = output.get('result', 'No result provided')
#     lines = result.split('\n')
#     #points = [line.strip() for line in lines if line.strip() and line.strip()[0].isdigit() and line.strip()[1:3] == ". "] 
#     result_formatted = result.replace("\n", "<br>")
#     formatted = (
#         "<div style='text-align: left; margin-left: 2em;'>"
#         "<h3 style='margin-bottom: 0.5em;'>Answer:</h3>"
#         f"<p style='margin-left: 1em;'>{result_formatted}</p>"
#         # f"<p style='margin-left: 1em;'>{result}</p>"
#         "</div>"
#     )
#     return formatted

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

def render_confusion_matrix_html() -> None:
    """Generates the confusion matrix HTML code as a string, preserving table layout."""
    y_true = st.session_state["eval_data"]["y_true"]
    y_pred = st.session_state["eval_data"]["y_pred"]

    # Calculate confusion matrix values.
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]

    # Calculate performance metrics.
    accuracy = accuracy_score(y_true, y_pred) if y_true and y_pred else 0.
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)
    specificity = TN / (TN + FP) if (TN + FP) else 0.

    # Build the HTML with embedded CSS, treating it as a column.
    html_code = f"""
      <style>
        .confusion-container {{
          background-color: #f3cac3;
          color: #000;
          padding: 10px;
          border-radius: 8px;
          border: 1px solid #333;
          font-family: Arial, sans-serif;
          width: 100%;
          max-width: 310px;           /* Slightly under sidebar width (~320px) */
          box-sizing: border-box;
          display: block;             /* Simple column layout */
        }}
        .confusion-container h2 {{
          margin: 0 0 10px 0;
          font-size: 1.3em;           /* Readable title */
          text-align: center;
        }}
        .stats {{
          margin: 0 0 10px 0;
          font-size: 0.9em;           /* Readable stats */
        }}
        .stats p {{
          margin: 3px 0;
          line-height: 1.2;           /* Ensure text is spaced */
        }}
        .table-container {{
          width: 100%;
          margin: 0 0 10px 0;
        }}
        .confusion-table {{
          border: 2px solid #000;
          border-collapse: collapse;
          text-align: center;
          width: 100%;
          table-layout: fixed;        /* Even column distribution */
          font-size: 0.9em;           /* Readable table text */
        }}
        .confusion-table th,
        .confusion-table td {{
          border: 1px solid #000;
          padding: 5px;
          word-wrap: break-word;      /* Wrap text to fit */
          vertical-align: middle;     /* Center content vertically */
        }}
        .confusion-table th {{
          background-color: #f8dcd7;
          white-space: normal;        /* Allow text to wrap */
        }}
        
      </style>
      <div class="confusion-container">
        <h2>Confusion Matrix</h2>
        <!-- Block container for the table -->
        <div class="table-container">
          <table class="confusion-table">
            <tr>
              <th></th>
              <th>Predicted True<br>(Detailed Answer)</th>
              <th>Predicted False<br>(Safe Disclaimer)</th>
            </tr>
            <tr>
              <th style="background-color: #f8dcd7;">Actual True<br>(Answerable)</th>
              <td>{TP} (TP)</td>
              <td>{FN} (FN)</td>
            </tr>
            <tr>
              <th style="background-color: #f8dcd7;">Actual False<br>(Unanswerable)</th>
              <td>{FP} (FP)</td>
              <td>{TN} (TN)</td>
            </tr>
          </table>
        </div>
        <div class="stats">
          <p><strong>Accuracy:</strong> {accuracy:.2f}</p>
          <p><strong>Precision:</strong> {precision:.2f}</p>
          <p><strong>Recall (Sensitivity):</strong> {sensitivity:.2f}</p>
          <p><strong>Specificity:</strong> {specificity:.2f}</p>
          <p><strong>F1 Score:</strong> {f1:.2f}</p>
        </div>
      </div>
    """
    st.html(html_code)

def add_feedback_buttons(response_content: str):
    """Adds copy, like, dislike, and speech buttons below the response."""
    feedback_script = f"""
    <script>
    function copyToClipboard(text, button) {{
        navigator.clipboard.writeText(text).then(function() {{
            // Change the icon to a checkmark
            button.innerHTML = '<i class="fas fa-check"></i>';
            // Revert back to the copy icon after 1.5 seconds
            setTimeout(function() {{
                button.innerHTML = '<i class="fas fa-copy"></i>';
            }}, 1500);
        }}, function(err) {{
            console.error('Failed to copy text: ', err);
        }});
    }}

    function handleFeedback(button, type) {{
        // Get the like and dislike buttons
        const likeButton = document.getElementById('like-button');
        const dislikeButton = document.getElementById('dislike-button');

        // Toggle the color of the clicked button
        if (button.style.color === 'red') {{
            // If already red, revert to white
            button.style.color = 'white';
        }} else {{
            // If not red, set to red and reset the other button
            button.style.color = 'red';
            if (type === 'like') {{
                dislikeButton.style.color = 'white';
            }} else {{
                likeButton.style.color = 'white';
            }}
        }}
    }}

    function toggleSpeech(button, text) {{
        // Check if speech is currently active
        if (button.style.color === 'red') {{
            // If red, stop speech and revert to white
            window.speechSynthesis.cancel();
            button.style.color = 'white';
            button.innerHTML = '<i class="fas fa-volume-up"></i>';
        }} else {{
            // If white, start speech and set to red
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
            style="
                background-color: gray; 
                color: white; 
                border: none; 
                padding: 8px; 
                border-radius: 50%; 
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            " 
            onclick="copyToClipboard(`{response_content}`, this)"
        >
            <i class="fas fa-copy"></i>
        </button>
        <button 
            id="like-button"
            style="
                background-color: gray; 
                color: white; 
                border: none; 
                padding: 8px; 
                border-radius: 50%; 
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            " 
            onclick="handleFeedback(this, 'like')"
        >
            <i class="fas fa-thumbs-up"></i>
        </button>
        <button 
            id="dislike-button"
            style="
                background-color: gray; 
                color: white; 
                border: none; 
                padding: 8px; 
                border-radius: 50%; 
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            " 
            onclick="handleFeedback(this, 'dislike')"
        >
            <i class="fas fa-thumbs-down"></i>
        </button>
        <button 
            id="speech-button"
            style="
                background-color: gray; 
                color: white; 
                border: none; 
                padding: 8px; 
                border-radius: 50%; 
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            " 
            onclick="toggleSpeech(this, `{response_content}`)"
        >
            <i class="fas fa-volume-up"></i>
        </button>
    </div>
    """
    components.html(feedback_script, height=40)

def updateEvalData(question: str, givenAnswer: str) -> None:
    questionIsTrulyAnswerable = question.strip() in ANSWERABLE_QUESTIONS
    questionIsPredictedAnswerable = any(keyword.lower() in givenAnswer[:ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK].lower() for keyword in CORRECT_ANSWER_KEYWORDS) or not any(keyword.lower() in givenAnswer[:ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK].lower() for keyword in UNANSWERABLE_ANSWER_KEYWORDS)

    st.session_state["eval_data"]["y_true"].append(questionIsTrulyAnswerable)
    st.session_state["eval_data"]["y_pred"].append(questionIsPredictedAnswerable)

def reset():
    st.session_state["cooldownBeginTimestamp"] = None
    st.session_state["messageTimes"] = []
    st.session_state["messages"] = []
    st.session_state["eval_data"] = {"y_true": [], "y_pred": []}
    st.session_state["reset"] = False

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

    st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")

    if "reset" not in st.session_state or st.session_state["reset"]:
        reset()

    with st.sidebar:
        matrix = st.empty()
        with matrix.container():
            render_confusion_matrix_html()

    primaryPage = st.empty()
    with primaryPage.container():
        for msg in st.session_state["messages"]:
            display_role = "A" if msg["role"] == "human" else msg["role"]
            with st.chat_message(display_role):
                st.markdown(msg["content"])
                if msg["role"] == "ai":
                    add_feedback_buttons(msg["content"])
                    

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error(f"To use the chatbot, please enter a Groq API key while running the launch script.")
            return

        class PlaceholderResponse():
            content = "[Example response]"

        if not DEBUG_MODE:
            # If no segments are provided, use default segments (ideally load from your JSON output)
            # if segments is None:
            #     segments = [
            #         "Example segment 1 text...",
            #         "Example segment 2 text...",
            #         # ... add more segments or load them dynamically from your scraped data.
            #     ]
            
            # 2. Build a FAISS vector store from the text segments.
            # vectorstore = FAISS.from_texts(segments, EMBEDDING_MODEL) if segments else None
            # vectorstore = FAISS(
            #     embedding_function=EMBEDDING_MODEL,
            #     index=faiss.read_index(INDEX_PATH),
            #     docstore=InMemoryDocstore(),
            #     index_to_docstore_id={}
            # )
            vectorstore = FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True) if INDEX_PATH is not None and os.path.isdir(INDEX_PATH) else None
            
            # 3. Instantiate your Groq API client using ChatGroq.
            ai = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=api_key,
            )
            
            # 4. Set up the RetrievalQA chain using the 'stuff' chain type.
            # qa_chain = RetrievalQA.from_chain_type(
            #     llm=ai,
            #     chain_type="stuff",
            #     retriever=vectorstore.as_retriever()
            # ) if vectorstore is not None else ai

        responseStartTime, responseEndTime = 0., 0.
        _count = 0

        # prompt = st.chat_input("What is your question?")
        prompts = ANSWERABLE_QUESTIONS[:(MAX_QUESTIONS_TO_ASK[0] if MAX_QUESTIONS_TO_ASK[0] else len(ANSWERABLE_QUESTIONS))] + UNANSWERABLE_QUESTIONS[:(MAX_QUESTIONS_TO_ASK[1] if MAX_QUESTIONS_TO_ASK[1] else len(UNANSWERABLE_QUESTIONS))]
        for prompt in prompts:
            if prompt and canAnswer():
                time.sleep(3)
                st.chat_message("A").markdown(prompt)
                st.session_state["messages"].append({"role": "human", "content": prompt})

                responseStartTime = time.monotonic()
                with st.chat_message("ai"):
                    if not DEBUG_MODE:
                        # Instead of directly invoking the ChatGroq API here,
                        # we call our retrieval chain function to get the domain-specific answer.
                        # TODO: Include previous message history in similarity search
                        context = str([doc.page_content for doc in vectorstore.similarity_search(prompt)]) if vectorstore is not None else ""
                        messages = [("system", SYSTEM_PROMPT + context)] + [(m["role"], m["content"]) for m in st.session_state["messages"]]
                        response = ai.invoke(messages)
                    else:
                        response = PlaceholderResponse()
                    responseEndTime = time.monotonic()
                    st.markdown(response.content)
                    st.session_state["messages"].append({"role": "ai", "content": response.content})
                    add_feedback_buttons(response.content)
                    responseTime = responseEndTime - responseStartTime
                    time_label = (
                        f":red[**{responseTime:.4f} seconds**]" 
                        if responseTime > MAX_RESPONSE_TIME 
                        else f"{responseTime:.4f} seconds"
                    )
                    st.markdown(f"*(Last response took {time_label})*")
                    
                updateEvalData(prompt, response.content)
                with st.sidebar:
                    with matrix.container():
                        render_confusion_matrix_html()
                        _count += 1
                        st.button("Reset", key=str(_count), on_click=reset, type="primary")
                scroll_to_bottom()

def main():
    """Entry point for the Streamlit app."""
    mainPage()

if __name__ == "__main__":
    main()
