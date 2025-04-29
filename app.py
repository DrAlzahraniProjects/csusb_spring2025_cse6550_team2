from apscheduler.schedulers.background import BackgroundScheduler
from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from urllib.parse import urlparse
import os
import re
import requests
import scrapy
import streamlit as st
import streamlit.components.v1 as components
import subprocess
import time
import hashlib
import json

URL_HASHES_PATH = "data/index/hashes.json"
URL_HASHES: dict[str, str] = {}

if os.path.exists(URL_HASHES_PATH):
    with open(URL_HASHES_PATH, "r") as f:
        try:
            URL_HASHES = json.load(f)
        except json.JSONDecodeError:
            URL_HASHES = {}

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

SYSTEM_PROMPT = f"""
You are Beta, an expert assistant for the Education Abroad program of California State University, San Bernardino (CSUSB).
You are designed to help students with all questions related to studying abroad.
You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.

Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student.
- **No Negative Responses:** Remain factual and avoid discouraging language.
- **Encourage and Inform:** Provide clear, supportive, and correct responses to the approved inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad (e.g., politics, religion, or personal debates).
- **Keep Responses Concise:** Limit your answers to 2–3 sentences to ensure brevity and clarity.
- **Website References Only:** If the answer is supported by retrieved context, include a 'References:' section listing all four source URLs from the context metadata. These will all be from the https://goabroad.csusb.edu domain.
  Example format:
  References:
  • https://goabroad.csusb.edu/page1
  • https://goabroad.csusb.edu/page2
  • https://goabroad.csusb.edu/page3
  • https://goabroad.csusb.edu/page4
- **No Fallback Links:** Do not refer to https://www.csusb.edu/studyabroad or any non-existent pages.
- **Do Not Add "Source:" or descriptions.** Only list the correct URL(s).
- **Do Not Guess:** If the provided context does not contain enough information to answer the question, respond only with: "I don't have enough information to answer this question."
"""

EMBEDDING_MODEL = FastEmbedEmbeddings()
RERANKER = Ranker(max_length=4096)
INDEX_PATH: str | None = os.path.join(".", "data", "index")
os.makedirs("data", exist_ok=True)

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def getInitialVectorstore() -> (FAISS | None):
    if DEBUG_MODE: return None
    try:
        return FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    except:
        return None

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = getInitialVectorstore()

TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')

class GoAbroadSpider(scrapy.Spider):
    name = "goabroad"
    allowed_domains = ["goabroad.csusb.edu"]
    start_urls = ["https://goabroad.csusb.edu/"]
    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 3,
    }

    def parse(self, response):
        global URL_HASHES
        global URL_HASHES_PATH
        raw_text_nodes = response.xpath("//body//text()[normalize-space()]").getall()
        joined_text = " ".join(text.strip() for text in raw_text_nodes if text.strip())
        cleaned_text = WHITESPACE_RE.sub(' ', TAG_RE.sub('', joined_text)).strip()
        content_hash = hash_text(cleaned_text)

        if URL_HASHES.get(response.url) != content_hash:
            URL_HASHES[response.url] = content_hash
            with open(URL_HASHES_PATH, "w") as f:
                json.dump(URL_HASHES, f, indent=2)

        segments = {cleaned_text[i:i + SEGMENT_SIZE].strip() for i in range(0, len(cleaned_text), SEGMENT_SIZE)}
        if "vectorstore" not in st.session_state or st.session_state["vectorstore"] is None: return
        st.session_state["vectorstore"].add_texts(segments, metadatas={"url": response.url})

        internal_links = response.css("a::attr(href)").getall()
        internal_links = list({response.urljoin(link) for link in internal_links if urlparse(response.urljoin(link)).hostname and "goabroad.csusb.edu" in urlparse(response.urljoin(link)).hostname})

        for link in internal_links:
            scrapy.Request(url=link, callback=self.parse)

def runScraper():
    subprocess.run(["scrapy", "crawl", "goabroad_spider"])

def launchAutomaticScraping():
    if st.session_state.get("automatic_scraping", False) or DEBUG_MODE: return
    scheduler = BackgroundScheduler()
    scheduler.add_job(runScraper, "interval", hours=24)
    scheduler.start()
    st.session_state["automatic_scraping"] = True

def scroll_to_bottom():
    components.html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)

def canAnswer() -> bool:
    currentTimestamp = time.monotonic()
    if st.session_state["cooldownBeginTimestamp"] is not None:
        if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
    else:
        st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:] + [currentTimestamp]
        if (len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or st.session_state["messageTimes"][-1] - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD):
            return True
        st.session_state["cooldownBeginTimestamp"] = currentTimestamp

    remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
    st.error(f"ERROR: Limit reached. You can resume in {int(remainingTime // 60)}m {int(remainingTime % 60)}s.")
    return False

def reset():
    st.session_state["cooldownBeginTimestamp"] = None
    st.session_state["messageTimes"] = []
    st.session_state["messages"] = []
    st.session_state["eval_data"] = {"y_true": [], "y_pred": []}
    st.session_state["reset"] = False

def rerank_results(question, documents):
    if not documents: return []
    pairs = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
    results = RERANKER.rerank(RerankRequest(question, pairs))
    ranked_docs = [documents[result["id"]] for result in results[:5]]
    return ranked_docs

def truncate_input(messages):
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
            st.error(f"Access to this webpage is prohibited.")
            st.stop()

    st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Education Abroad Chatbot</h1>")
    st.html("<p align='center'>This is a chatbot for answering questions about CSUSB's Education Abroad program, based on <a href='https://goabroad.csusb.edu'>goabroad.csusb.edu</a>.</p>")

    if st.session_state.get("reset", True):
        reset()

    for msg in st.session_state["messages"]:
        display_role = "human" if msg["role"] == "human" else msg["role"]
        with st.chat_message(display_role):
            st.markdown(msg["content"])

    api_key = os.environ["GROQ_API_KEY"] 
    if not api_key:
        st.error("Missing Groq API Key.")
        st.stop()

    class PlaceholderResponse():
        content = "[Example response]"

    if not DEBUG_MODE:
        ai = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=None, timeout=None, max_retries=2, api_key=api_key)

    user_input = st.chat_input("Ask about studying abroad from CSUSB...")

    if user_input and canAnswer():
        with st.chat_message("human"):
            st.markdown(user_input)
            st.session_state["messages"].append({"role": "human", "content": user_input})

        responseStartTime = time.monotonic()
        with st.chat_message("ai"):
            try:
                initial_docs = st.session_state["vectorstore"].similarity_search(user_input, k=100)
                ranked_docs=rerank_results(user_input,initial_docs)
                url_to_doc = {}
                for doc in ranked_docs:
                    url = doc.metadata.get("url", "").strip()
                    if url.startswith("https://goabroad.csusb.edu") and url not in url_to_doc:
                        url_to_doc[url] = doc
                    # if len(url_to_doc) == 4:
                    #     break

                final_urls = list(url_to_doc.keys())
                final_segments = [doc.page_content[:500] for doc in url_to_doc.values()]

                context = " ".join(final_segments)
                messages = [("system", SYSTEM_PROMPT + context)] + [(m["role"], m["content"]) for m in st.session_state["messages"][-MAX_HISTORY_TO_USE:]]
                truncated_messages = truncate_input(messages)
                response = ai.invoke(truncated_messages)

                response.content = re.sub(r"(?i)(references:.*?)(\n\n|\Z)", "", response.content, flags=re.DOTALL).strip()

                if final_urls and response.content.strip() != "I don't have enough information to answer this question.":
                    response.content += "\n\nReferences:\n" + "\n".join(f"• [Source {i}]({url})" for i, url in enumerate(final_urls))

            except Exception as e:
                st.error(f"Error generating Beta's response: {e}")
                response = PlaceholderResponse()

            responseEndTime = time.monotonic()
            st.markdown(response.content)
            st.session_state["messages"].append({"role": "ai", "content": response.content})
            st.markdown(f"*(Last response took {responseEndTime - responseStartTime:.4f} seconds)*")

        scroll_to_bottom()

def main():
    mainPage()
    launchAutomaticScraping()

if __name__ == "__main__":
    main()
