from apscheduler.schedulers.background import BackgroundScheduler
# from faiss import IndexFlatL2
from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.docstore.in_memory import InMemoryDocstore
from urllib.parse import urlparse
import os
import re
import requests
import scrapy
import streamlit as st
import streamlit.components.v1 as components
import subprocess
import time
import uuid
import hashlib
import pathlib
import json

URL_HASHES_PATH = "data/index/hashes.json"
URL_HASHES: dict[str, str] = {}

# Load saved hashes at startup
if os.path.exists(URL_HASHES_PATH):
    with open(URL_HASHES_PATH, "r") as f:
        try:
            URL_HASHES = json.load(f)
        except json.JSONDecodeError:
            URL_HASHES = {}


# Constants
RESTRICT_IP: bool = True
COOLDOWN_CHECK_PERIOD = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN = 10
COOLDOWN_DURATION = 180.0
MAX_RESPONSE_TIME = 3.0
ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK = 30
MAX_AI_INPUT_CHARACTERS: int = 5000
MAX_HISTORY_TO_USE: int = 8
DEBUG_MODE: bool = False
SEGMENT_SIZE: int = 512

# Updated system prompt for Beta
SYSTEM_PROMPT = """
You are Beta, an expert assistant for the Study Abroad program of California State University, San Bernardino (CSUSB).
You are designed to help students with all questions related to studying abroad.
You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.

Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student.
- **No Negative Responses:** Remain factual and avoid discouraging language.
- **Encourage and Inform:** Provide clear, supportive, and correct responses to the approved inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad (e.g., politics, religion, or personal debates).
- **Keep Responses Concise:** Limit your answers to 2-3 sentences to ensure brevity and clarity.

Provide a concise and accurate answer based solely on the context below.
If the context does not contain enough information to answer the question, respond with "I don't have enough information to answer this question." Do not generate, assume, or make up any details beyond the given context.
"""


RANDOM_QUESTION_PROMPT = {"answerable": """
You are an AI designed to generate diverse questions about studying abroad at CSUSB.
Generate a random question about CSUSB study abroad programs, scholarships, visa processes, or cultural adaptation, that is answerable by the provided context.
Ensure the question is unique and does not duplicate any previous questions listed below. Do not return any output besides the question itself.
""", "unanswerable": """
You are an AI designed to generate diverse questions.
Generate a random question pertaining to any topic EXCEPT university study abroad programs. Any other domain of knowledge is allowed.
Ensure the question is unique and does not duplicate any previous questions listed below. Do not return any output besides the question itself.
"""}

ANSWERABLE_QUESTIONS: tuple[str, ...] = (
    "What study abroad programs are offered through CSUSB?",
    "Are there specific scholarships for CSUSB students studying abroad?",
    "How can I find partner universities for direct enrollment through CSUSB?",
    "Does CSUSB provide assistance with obtaining a visa for studying abroad?",
    "Can I study abroad in a country where English is not the primary language?"
)
UNANSWERABLE_QUESTIONS: tuple[str, ...] = (
    "What is the meaning of life, and how does studying abroad contribute to it?",
    "If a tree falls in a foreign country and no one is around to hear it, does it make a sound?",
    "What will the world look like in 1,000 years, and how will study abroad programs evolve?",
    "How do you reconcile the existence of suffering with the pursuit of global education?",
    "Can you prove that reality is not a simulation, and if it is, how does studying abroad fit into it?"
)
CORRECT_ANSWER_KEYWORDS: tuple[str, ...] = ("yes", "indeed", "correct", "right")
UNANSWERABLE_ANSWER_KEYWORDS: tuple[str, ...] = ("cannot answer", "can't answer", "cannot help with", "cannot help you with", "can't help with", "can't help you with", "do not know", "don't know", "do not have enough info", "don't have enough info", "not knowledgable", "please refer", "don't have access", "do not have access", "cannot access", "can't access")

# Initialize models
# EMBEDDING_MODEL = OllamaEmbeddings(model="llama3")
EMBEDDING_MODEL = FastEmbedEmbeddings()
RERANKER = Ranker(max_length=4096)
INDEX_PATH: str | None = os.path.join(".", "data", "index")

os.makedirs("data", exist_ok=True)


def hash_text(text: str) -> str:
    """Return MD5 hash of a given text"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# def get_cache_path(url: str) -> pathlib.Path:
#     """Return file path for cache based on URL hash"""
#     hashed = hashlib.md5(url.encode("utf-8")).hexdigest()
#     return pathlib.Path("data/cache") / f"{hashed}.txt"


def getInitialVectorstore() -> (FAISS | None):
    if DEBUG_MODE: return None
    try:
        return FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    except:
        # return FAISS(EMBEDDING_MODEL, IndexFlatL2(SEGMENT_SIZE), InMemoryDocstore(), {})
        return None

if "vectorstore" not in st.session_state:
    # vectorstore = getInitialVectorstore()
    # st.session_state["vectorstoreInitialized"] = True
    st.session_state["vectorstore"] = getInitialVectorstore()

TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')
class GoAbroadSpider(scrapy.Spider):
    name = "goabroad"
    allowed_domains = ["goabroad.csusb.edu"]
    start_urls = ["https://goabroad.csusb.edu/"]

    # Custom settings for politeness.
    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 3,
    }

    def parse(self, response):
        global URL_HASHES
        global URL_HASHES_PATH
        # self.logger.info(f"Parsing URL: {response.url}")
        # Gather reference information.
        # url = response.url
        # title = response.xpath("//title/text()").get(default="").strip()
        # meta_description = response.xpath("//meta[@name='description']/@content").get(default="").strip()

        # Extract structured data (e.g., JSON‑LD).
        # structured_data = response.xpath("//script[@type='application/ld+json']/text()").getall()

        # Extract text nodes from the body of the page.
        raw_text_nodes = response.xpath("//body//text()[normalize-space()]").getall()
        joined_text = " ".join(text.strip() for text in raw_text_nodes if text.strip())

        # Clean the text.
        cleaned_text = WHITESPACE_RE.sub(' ', TAG_RE.sub('', joined_text)).strip()
        
        content_hash = hash_text(cleaned_text)

        if URL_HASHES.get(response.url) == content_hash:
            self.logger.info(f"[SKIPPED] No change for {response.url}")
        else:
            # Update in-memory cache
            URL_HASHES[response.url] = content_hash

            # Persist the updated hash table to disk
            with open(URL_HASHES_PATH, "w") as f:
                json.dump(URL_HASHES, f, indent=2)

        # Segment the cleaned text into chunks.
        segments = {cleaned_text[i:i + SEGMENT_SIZE].strip() for i in range(0, len(cleaned_text), SEGMENT_SIZE)}

        # while ("vectorstore" not in globals()) or not isinstance(vectorstore, FAISS): time.sleep(30)
        if "vectorstore" not in st.session_state or st.session_state["vectorstore"] is None: return
        st.session_state["vectorstore"].add_texts(segments, metadatas={"url": response.url})

        # Extract and normalize internal links for further crawling.
        internal_links = response.css("a::attr(href)").getall()
        internal_links = list({response.urljoin(link) for link in internal_links if urlparse(response.urljoin(link)).hostname is not None and (urlparse(response.urljoin(link)).hostname == "goabroad.csusb.edu" or urlparse(response.urljoin(link)).hostname.endswith(".goabroad.csusb.edu"))})

        # yield {
        #     "url": url,
        #     # "title": title,
        #     # "meta_description": meta_description,
        #     # "structured_data": structured_data,
        #     # "cleaned_text": cleaned_text,
        #     "segments": segments,
        #     # "internal_links": internal_links,
        # }

        # Follow internal links to continue crawling the site.
        for link in internal_links:
            # yield scrapy.Request(url=link, callback=self.parse)
            scrapy.Request(url=link, callback=self.parse)

def runScraper():
    subprocess.run(["scrapy", "crawl", "goabroad_spider"])

def launchAutomaticScraping():
    if st.session_state.get("automatic_scraping", False) or DEBUG_MODE: return
    time.sleep(10)
    st.write("[Launching scraping worker]")
    scheduler = BackgroundScheduler()
    scheduler.add_job(runScraper, "interval", hours=24)
    scheduler.start()
    st.session_state["automatic_scraping"] = True

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
    # If a cooldown exists:
    if st.session_state["cooldownBeginTimestamp"] is not None:
        # And the duration has already elapsed, no problem exists
        if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
        # Case of duration not having elapsed falls through
    else:
        # Track last N message times. If < N messages have been sent or time between current and Nth message is above cooldown, no problem exists
        st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:] + [currentTimestamp]
        if (len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or st.session_state["messageTimes"][-1] - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD):
            return True
        # Set timestamp of cooldown beginning
        st.session_state["cooldownBeginTimestamp"] = currentTimestamp

    cooldownMinutes = int(COOLDOWN_CHECK_PERIOD // 60)
    cooldownSeconds = int(COOLDOWN_CHECK_PERIOD) % 60
    remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
    remainingMinutes = int(remainingTime // 60)
    remainingSeconds = int(remainingTime) % 60
    st.error(
        f"ERROR: The app has reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} question{'s' if MAX_MESSAGES_BEFORE_COOLDOWN != 1 else ''} per "
        f"{cooldownMinutes} minute{'s' if cooldownMinutes != 1 else ''} {cooldownSeconds} second{'s' if cooldownSeconds != 1 else ''}. "
        f"You can resume in {remainingMinutes} minute{'s' if remainingMinutes != 1 else ''} "
        f"{remainingSeconds} second{'s' if remainingSeconds != 1 else ''}."
    )
    return False

def update_like(response_id):
    """Update TP when user likes the response."""
    if "feedback_data" not in st.session_state:
        st.session_state["feedback_data"] = {}

    current_feedback = st.session_state["feedback_data"].get(response_id, None)

    if current_feedback == "liked":
        st.session_state["feedback_data"][response_id] = None
    else:
        st.session_state["feedback_data"][response_id] = "liked"
        st.session_state["eval_data"]["y_true"].append(True)
        st.session_state["eval_data"]["y_pred"].append(True)

def update_unlike(response_id):
    """Update FN when user dislikes the response."""
    if "feedback_data" not in st.session_state:
        st.session_state["feedback_data"] = {}

    current_feedback = st.session_state["feedback_data"].get(response_id, None)

    if current_feedback != "disliked":
        st.session_state["feedback_data"][response_id] = "disliked"
        st.session_state["eval_data"]["y_true"].append(True)
        st.session_state["eval_data"]["y_pred"].append(False)

def copy_response(text):
    """Copy the response text to the clipboard using JavaScript."""
    copy_script = f"""
    <script>
    try {{
        navigator.clipboard.writeText(`{text}`).then(function() {{
            console.log('Text copied to clipboard');
        }}).catch(function(err) {{
            console.error('Failed to copy text: ', err);
            alert('Failed to copy text: ' + err);
        }});
    }} catch (err) {{
        console.error('Error in copy script: ', err);
        alert('Error in copy script: ' + err);
    }}
    </script>
    """
    components.html(copy_script, height=0)

def speak_response(text, agentIndex):
    """Enhanced to support real-time audio streaming"""
    speech_script = f"""
    <script>
    if ('speechSynthesis' in window) {{
        const availableVoices = window.speechSynthesis.getVoices();
        const utterance = new SpeechSynthesisUtterance(`{text}`);
        utterance.onstart = () => console.log('Speech started');
        utterance.onend = () => console.log('Speech ended');
        utterance.rate = 1 + Math.random()*3.0/10.0;
        if (availableVoices.length) utterance.voice = availableVoices[Math.min({agentIndex}, availableVoices.length - 1)];
        window.speechSynthesis.speak(utterance);
    }} else {{
        console.error('Web Speech API not supported');
    }}
    </script>
    """
    components.html(speech_script, height=0)

def render_confusion_matrix_html() -> None:
    """Generates the confusion matrix HTML code as a string, preserving table layout."""
    y_true = st.session_state["eval_data"]["y_true"]
    y_pred = st.session_state["eval_data"]["y_pred"]

    TP = sum(t & p for t, p in zip(y_true, y_pred, strict=True))
    FN = sum(t & (not p) for t, p in zip(y_true, y_pred, strict=True))
    FP = sum((not t) & p for t, p in zip(y_true, y_pred, strict=True))
    TN = sum((not t) & (not p) for t, p in zip(y_true, y_pred, strict=True))

    accuracy = (TP + TN)/len(y_true) if y_true else 0.
    precision = TP/(TP + FP) if TP + FP else 0.
    sensitivity = TP/(TP + FN) if TP + FN else 0.
    f1 = 2*TP/(2*TP + FN + FP) if 2*TP + FN + FP else 0.
    specificity = TN/(TN + FP) if TN + FP else 0.

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
          max-width: 310px;
          box-sizing: border-box;
          display: block;
        }}
        .confusion-container h2 {{
          margin: 0 0 10px 0;
          font-size: 1.3em;
          text-align: center;
        }}
        .stats {{
          margin: 0 0 10px 0;
          font-size: 0.9em;
        }}
        .stats p {{
          margin: 3px 0;
          line-height: 1.2;
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
          table-layout: fixed;
          font-size: 0.9em;
        }}
        .confusion-table th,
        .confusion-table td {{
          border: 1px solid #000;
          padding: 5px;
          word-wrap: break-word;
          vertical-align: middle;
        }}
        .confusion-table th {{
          background-color: #f8dcd7;
          white-space: normal;
        }}
      </style>
      <div class="confusion-container">
        <h2>Confusion Matrix</h2>
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

def add_feedback_buttons(response_content: str, response_id: str, agentIndex: int):
    """Displays Copy, Like, Dislike, and Speech buttons in one row."""
    if not response_id:
        response_id = str(uuid.uuid4())

    st.markdown(
        """
        <style>
        .stButton button {
            border-radius: 50%;
            width: 40px;
            height: 40px;
            padding: 0;
            margin: 0 5px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    
    if col1.button("📋", key=f"copy_{response_id}"):
        copy_response(response_content)
    
    if "feedback_data" in st.session_state and st.session_state["feedback_data"].get(response_id) == "liked":
        col2.button("👍", key=f"like_{response_id}", on_click=update_like, args=(response_id,))
    else:
        col2.button("👍", key=f"like_{response_id}", on_click=update_like, args=(response_id,))
    
    if "feedback_data" in st.session_state and st.session_state["feedback_data"].get(response_id) == "disliked":
        col3.button("👎", key=f"dislike_{response_id}", on_click=update_unlike, args=(response_id,))
    else:
        col3.button("👎", key=f"dislike_{response_id}", on_click=update_unlike, args=(response_id,))
    
    col4.button("🔊", key=f"speech_{response_id}", on_click=speak_response, args=(response_content, agentIndex))

def updateEvalData(question: str, givenAnswer: str) -> None:
    """Update evaluation data when Beta AI generates an answer."""
    questionIsTrulyAnswerable = question.strip() in ANSWERABLE_QUESTIONS
    questionIsPredictedAnswerable = any(
        keyword.lower() in givenAnswer[:ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK].lower()
        for keyword in CORRECT_ANSWER_KEYWORDS
    ) or not any(
        keyword.lower() in givenAnswer[:ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK].lower()
        for keyword in UNANSWERABLE_ANSWER_KEYWORDS
    )

    st.session_state["eval_data"]["y_true"].append(questionIsTrulyAnswerable)
    st.session_state["eval_data"]["y_pred"].append(questionIsPredictedAnswerable)

def reset():
    st.session_state["cooldownBeginTimestamp"] = None
    st.session_state["messageTimes"] = []
    st.session_state["messages"] = []
    st.session_state["eval_data"] = {"y_true": [], "y_pred": []}
    stop_all_speech_script = """
    <script>
    if ('speechSynthesis' in window) {
        window.speechSynthesis.pause();
        window.speechSynthesis.cancel();
    }
    </script>
    """
    components.html(stop_all_speech_script, height=0)
    st.session_state["reset"] = False

def rerank_results(question, documents):
    """Rerank search results using FlashRank without comparing Document objects directly."""
    if not documents:
        return []
    
    # Create pairs for FlashRank
    pairs = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
    # Get sorted pairs from FlashRank
    results = RERANKER.rerank(RerankRequest(question, pairs))
    # Reorder documents based on sorted indices, taking top 5
    ranked_docs = [result["text"] for result in results[:5]]
    return ranked_docs

def estimate_tokens(text):
    """Roughly estimate token count based on word count (1 word ≈ 1 token)"""
    return len(text.split())

def truncate_input(messages):
    """Truncate the combined input messages to a maximum of MAX_AI_INPUT_CHARACTERS characters."""
    combined_text = []
    for msg in reversed(messages):
        msg_length = sum(len(part) for part in msg)
        if len(combined_text) + msg_length > MAX_AI_INPUT_CHARACTERS:
            break
        combined_text.append(msg)
    combined_text.reverse()
    return combined_text


def get_user_ip() -> str:
    try:
        # When Streamlit is running inside a container, the Request object is not accessible, so this method cannot be used to get the public IP
        response = requests.get('https://api.ipify.org?format=json', timeout=None)
        return response.json().get("ip", "")
    except Exception:
        return ""

def is_csusb_ip(ip: str) -> bool:
    return any([
        ip.startswith("138.23."),
        ip.startswith("139.182."),
        ip.startswith("152.79.") 
    ])


def mainPage():
    if RESTRICT_IP:
        user_ip = get_user_ip()
        if not is_csusb_ip(user_ip):
            st.error(f"Access denied: Your IP ({user_ip}) is not from CSUSB campus network.")
            st.stop()
        else:
            st.warning(f"Your IP is part of the CSUSB campus network and so has been allowed.")

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

    # Display chat history
    primaryPage = st.empty()
    with primaryPage.container():
        # Real-time conversation container
        # conversation_container = st.empty()
         for msg in st.session_state["messages"]:
            display_role = "human" if msg["role"] == "human" else msg["role"]
            with st.chat_message(display_role):
                st.markdown(msg["content"])
            if msg["role"] == "ai":
                add_feedback_buttons(msg["content"], msg.get("id", ""), 1)

    # Load vectorstore and model
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error(f"To use the chatbot, please enter a Groq API key while running the launch script.")
        st.stop()

    class PlaceholderResponse():
        content = "[Example response]"

    if not DEBUG_MODE:
        # vectorstore = FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True) if INDEX_PATH is not None and os.path.isdir(INDEX_PATH) else None
        # vectorstore = getInitialVectorstore()
        ai = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key,
        )

    # === ✅ USER INPUT SECTION ===
    user_input = st.chat_input("Ask about studying abroad at CSUSB...")

    if user_input and canAnswer():
        with st.chat_message("human"):
            st.markdown(user_input)
            # speak_response(user_input, 0)
            st.session_state["messages"].append({"role": "human", "content": user_input})

        responseStartTime = time.monotonic()
        with st.chat_message("ai"):
            try:
                initial_docs = st.session_state["vectorstore"].similarity_search(user_input) if "vectorstore" in st.session_state and st.session_state["vectorstore"] else []
                ranked_docs = rerank_results(user_input, initial_docs)
                context = " ".join([doc[:500] if isinstance(doc, str) else doc.page_content[:500] for doc in ranked_docs])
                messages = [("system", SYSTEM_PROMPT + context)] + [(m["role"], m["content"]) for m in st.session_state["messages"][-MAX_HISTORY_TO_USE:]]
                truncated_messages = truncate_input(messages)
                response = ai.invoke(truncated_messages)
            except Exception as e:
                st.error(f"Error generating Beta's response: {e}")
                response = PlaceholderResponse()

            responseEndTime = time.monotonic()
            responseTime = responseEndTime - responseStartTime
            response_id = str(uuid.uuid4())

            st.markdown(response.content)
            # speak_response(response.content, 1)
            st.session_state["messages"].append({"role": "ai", "content": response.content})
            add_feedback_buttons(response.content, response_id, 1)

            st.markdown(f"*(Last response took {responseTime:.4f} seconds)*")

        updateEvalData(user_input, response.content)

        with st.sidebar:
            with matrix.container():
                render_confusion_matrix_html()
                st.button("Reset", key=str(uuid.uuid4()), on_click=reset, type="primary")

        scroll_to_bottom()



def main():
    mainPage()
    launchAutomaticScraping()

if __name__ == "__main__":
    main()
