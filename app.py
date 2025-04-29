from apscheduler.schedulers.background import BackgroundScheduler
# from datetime import date
# from faiss import IndexFlatL2
from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document # Import Document type
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
import hashlib
import json
from cachetools import TTLCache
import hashlib
import math
from typing import Tuple, Dict, Any, List

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
RESTRICT_IP: bool = False
COOLDOWN_CHECK_PERIOD = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN = 10
COOLDOWN_DURATION = 180.0
ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK = 30
MAX_AI_INPUT_CHARACTERS: int = 5000
MAX_HISTORY_TO_USE: int = 8
DEBUG_MODE: bool = False
SEGMENT_SIZE: int = 512
SEMANTIC_SIMILARITY_THRESHOLD = 0.95  # Adjust based on testing
CACHE_ENTRY = Tuple[list[float], str]  # Type alias: (embedding, answer)
MAX_RESPONSE_TIME = 3.0
TIMEOUT_MESSAGE = "Request timeout: The response took too long to generate. Please try again with a more specific question."

# --- MODIFIED SYSTEM PROMPT ---
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
# The current date is {date.today().strftime('%A, %B %d, %Y')}.


# Initialize models
EMBEDDING_MODEL = FastEmbedEmbeddings()
RERANKER = Ranker(max_length=4096)
INDEX_PATH: str | None = os.path.join(".", "data", "index")

os.makedirs("data", exist_ok=True)

# Modify the cache initialization to store embeddings
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = TTLCache(maxsize=100, ttl=3600)
ANSWER_CACHE: Dict[str, CACHE_ENTRY] = st.session_state.answer_cache  # Now stores (embedding, answer)

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors using pure Python"""
    dot_product = sum(a*b for a,b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a*a for a in vec_a))
    norm_b = math.sqrt(sum(b*b for b in vec_b))
    return dot_product / (norm_a * norm_b + 1e-10)  # Small epsilon to avoid division by zero

def find_semantic_match(user_input: str) -> Tuple[str, float] | None:
    """
    Check cache for semantically similar questions.
    Returns (cached_answer, similarity_score) if found, else None.
    """
    if not ANSWER_CACHE:
        return None

    input_embedding = EMBEDDING_MODEL.embed_query(user_input)
    best_match = None
    highest_sim = 0.0

    # Iterate over items to get both key (original question/hash) and value (embedding, answer)
    for original_input, (cached_embedding, cached_answer) in ANSWER_CACHE.items():
        sim = cosine_similarity(input_embedding, cached_embedding)
        if sim > highest_sim and sim >= SEMANTIC_SIMILARITY_THRESHOLD:
            highest_sim = sim
            best_match = cached_answer

    return (best_match, highest_sim) if best_match else None


def generate_md5_hash(question: str) -> str:
    """Generate MD5 hash for any input text"""
    return hashlib.md5(question.encode("utf-8")).hexdigest()

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
        cleaned_text = WHITESPACE_RE.sub(' ', TAG_RE.sub('', joined_text)).strip()

        content_hash = generate_md5_hash(cleaned_text)

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
        # Add segments with metadata to the vectorstore
        # Convert segments (strings) back to Document objects to preserve metadata capability
        docs_to_add = [Document(page_content=segment, metadata={"url": response.url}) for segment in segments]
        st.session_state["vectorstore"].add_documents(docs_to_add)


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
    if st.session_state.get("cooldownBeginTimestamp") is not None:
        # And the duration has already elapsed, no problem exists
        if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
        # Case of duration not having elapsed falls through
    else:
        # Track last N message times. If < N messages have been sent or time between current and Nth message is above cooldown, no problem exists
        st.session_state["messageTimes"] = st.session_state.get("messageTimes", [])[-MAX_MESSAGES_BEFORE_COOLDOWN:] + [currentTimestamp]
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

# Define the chat interaction method here
def handle_chat_interaction(ai: ChatGroq | None):
    """Handles user input, AI response generation, and displaying chat messages."""
    # === USER INPUT SECTION ===
    user_input = st.chat_input("Ask about studying abroad from CSUSB...")

    if user_input and canAnswer():
        with st.chat_message("human"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "human", "content": user_input})

        responseStartTime = time.monotonic()

        # Use a placeholder or initial message in the AI bubble
        with st.chat_message("ai"):
            message_placeholder = st.empty() # Create an empty element to progressively update
            full_response = "" # Accumulate the full response for caching

            cache_key = generate_md5_hash(user_input)

            # 1. First try exact cache match
            cached_response_content = None
            if cache_key in ANSWER_CACHE:
                cached_embedding, cached_response_content = ANSWER_CACHE[cache_key]
                full_response = cached_response_content
                message_placeholder.markdown(full_response)


            # 2. Check for semantic matches if no exact cache hit and no exact cached response
            if not cached_response_content:
                semantic_match = find_semantic_match(user_input)
                if semantic_match:
                    cached_response_content, _ = semantic_match
                    full_response = cached_response_content
                    message_placeholder.markdown(full_response)


            # 3. Fallback to API call and STREAMING if no cache hits or semantic matches
            if not cached_response_content:
                 try:
                    # The timeout is now handled by the ChatGroq instance
                    initial_docs = []
                    if st.session_state.get("vectorstore", None):
                        initial_docs = st.session_state["vectorstore"].similarity_search(user_input, k=100) # Get more documents initially

                    # Use the modified rerank_results that returns Document objects
                    ranked_docs = rerank_results(user_input, initial_docs)

                    # --- CITATION LOGIC START ---
                    url_to_doc = {}
                    # Filter and collect goabroad.csusb.edu URLs from ranked documents
                    for doc in ranked_docs:
                        # Ensure doc is a Document object and has metadata
                        if isinstance(doc, Document) and doc.metadata and "url" in doc.metadata:
                            url = doc.metadata.get("url", "").strip()
                            # Only include goabroad.csusb.edu URLs
                            if url.startswith("https://goabroad.csusb.edu"):
                                # We still add to url_to_doc to get one Document object per URL,
                                # useful if we needed to retrieve original doc details later,
                                # but the uniqueness check will be on the list derived from keys.
                                url_to_doc[url] = doc

                    # Get the list of URLs from the dictionary keys
                    potential_final_urls = list(url_to_doc.keys())

                    # --- Explicitly ensure URLs are unique using a set ---
                    unique_final_urls = []
                    seen_urls = set()
                    for url in potential_final_urls:
                        if url not in seen_urls:
                            unique_final_urls.append(url)
                            seen_urls.add(url)
                

                    # Use the page content from the selected unique documents for context
                    # Retrieve the Document objects corresponding to the unique URLs to get their content
                    final_segments = [url_to_doc[url].page_content[:500] for url in unique_final_urls if url in url_to_doc]

                    # Construct context from the content of the selected documents
                    context = " ".join(final_segments) if final_segments else ""
                    # --- CITATION LOGIC END ---

                    # Add context to the system prompt
                    messages = [("system", SYSTEM_PROMPT + "\n\nContext:\n" + context)] + [(m["role"], m["content"]) for m in st.session_state["messages"][-MAX_HISTORY_TO_USE:]]
                    truncated_messages = truncate_input(messages)

                    # === STREAMING IMPLEMENTATION ===
                    # Use the .stream() method provided by ChatGroq
                    if ai: # Check if ai model was initialized and passed
                        raw_response_content = "" # Store AI's raw response before adding references
                        for chunk in ai.stream(truncated_messages):
                            if chunk.content is not None:
                                raw_response_content += chunk.content # Accumulate chunks
                                # Display chunk and a typing indicator (optional, but good for UX)
                                message_placeholder.markdown(raw_response_content + "▌")
                        full_response = raw_response_content # The full response from the AI API

                        # --- Append References ---
                        # Append references if unique URLs were found and the AI didn't respond with the "not enough information" message
                        if unique_final_urls and full_response.strip() != "I don't have enough information to answer this question.":
                             # Remove any existing "References:" section the model might have hallucinated
                            full_response = re.sub(r"(?i)(references:.*?)(\n\n|\Z)", "", full_response, flags=re.DOTALL).strip()
                            # Append the correctly formatted references using unique_final_urls
                            full_response += "\n\nReferences:\n" + "\n".join(f"• [Source {i+1}]({url})" for i, url in enumerate(unique_final_urls)) # Source 1, 2, etc.

                        message_placeholder.markdown(full_response) # Display final complete response with references
                    else:
                         # This case should ideally not happen if the AI is initialized in mainPage,
                         # but as a fallback/in DEBUG_MODE
                        full_response = "AI model is not available."
                        message_placeholder.markdown(full_response)
                    # === END STREAMING IMPLEMENTATION ===

                    # Store in cache after streaming is complete, only if it came from the API
                    if ai: # Only cache if the AI model was used
                        embedding = EMBEDDING_MODEL.embed_query(user_input)
                        # Cache the *final* response including references
                        ANSWER_CACHE[cache_key] = (embedding, full_response)


                 except Exception as e:
                    # Catch timeout specifically if the library raises a specific exception
                    if "timeout" in str(e).lower():
                        full_response = TIMEOUT_MESSAGE
                    else:
                        # Ensure full_response is set even on other errors before displaying
                        full_response = f"Error generating response: {str(e)}"
                        st.error(full_response) # Keep the st.error for visibility outside the placeholder if needed
                    message_placeholder.markdown(full_response) # Display the error or timeout message within the placeholder

            # This part was outside the if not cached_response_content block before.
            # It should be here to ensure messages are appended whether from cache or API.
            st.session_state["messages"].append({"role": "ai", "content": full_response})

            responseEndTime = time.monotonic()
            st.markdown(f"*(Last response took {responseEndTime - responseStartTime:.4f} seconds)*")


def mainPage():
    if RESTRICT_IP:
        user_ip = get_user_ip()
        if not is_csusb_ip(user_ip):
            st.error(f"Access to this webpage is prohibited.")
            st.stop()

    st.html("""
        <style>
            body {
                background-color: #007BFF !important;
                color: white !important;
            }
        </style>
    """)

    st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Education Abroad Chatbot</h1>")
    st.html("<p align=\"center\">This is a chatbot for answering questions about CSUSB's Education Abroad program, based on the details from its website (<a href=\"https://goabroad.csusb.edu\">goabroad.csusb.edu</a>).</p>")

    if st.session_state.get("reset", True):
        reset()

    # Display chat history
    primaryPage = st.empty()
    with primaryPage.container():
        # Real-time conversation container
        for msg in st.session_state["messages"]:
            display_role = "human" if msg["role"] == "human" else msg["role"]
            with st.chat_message(display_role):
                st.markdown(msg["content"])

    # Load vectorstore and model
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error(f"To use the chatbot, please enter a Groq API key while running the launch script.")
        st.stop()

    # Initialize AI model - This remains in mainPage as per your instruction
    ai = None
    if not DEBUG_MODE:
        ai = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=None,
            timeout=None, # Apply timeout here for the API call itself
            max_retries=2,
            api_key=api_key,
        )

    # Call the new function to handle chat interaction, passing the initialized AI model
    handle_chat_interaction(ai)

    scroll_to_bottom()


def main():
    mainPage()
    launchAutomaticScraping()

if __name__ == "__main__":
    main()