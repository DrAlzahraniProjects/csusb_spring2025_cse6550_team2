from cachetools import TTLCache
from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from math import sqrt
from typing import Iterable
import hashlib
import os
import requests
import streamlit as st
import streamlit.components.v1 as components
import time

# =======================
# IP RESTRICTION (U.S. only)
# =======================
def get_user_ip() -> str:
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=None)
        return response.json().get("ip", "")
    except Exception:
        return ""

def is_US_ip(ip: str) -> bool:
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}?fields=country", timeout=None)
        return response.json().get("country", "").lower() == "united states"
    except Exception:
        return False

user_ip = get_user_ip()
if not is_US_ip(user_ip):
    st.error("Access to this webpage is prohibited.")
    st.stop()

# =======================
# Constants and Setup
# =======================
COOLDOWN_CHECK_PERIOD: float = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN: int = 10
COOLDOWN_DURATION: float = 180.0
ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK: int = 30
MAX_AI_INPUT_CHARACTERS: int = 5000
MAX_HISTORY_TO_USE: int = 8
SEGMENT_SIZE: int = 512
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.95
CONTEXT_RELEVANCE_THRESHOLD: float = 0.55 # Could use tuning. < 0.45 seems to be too low. > 0.65 seems to be too high.

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
- **Keep Responses Concise:** Limit your answers to 2-3 sentences to ensure brevity and clarity.

Provide a concise and accurate answer based solely on the context below.
If the context does not contain enough information to answer the question, respond with "I don't have enough information to answer this question." Do not generate, assume, or make up any details beyond the given context.
"""

EMBEDDING_MODEL = FastEmbedEmbeddings()
RERANKER = Ranker(max_length=4096)
INDEX_PATH: str | None = os.path.join(".", "data", "index")

# =======================
# Vector Search and Similarity
# =======================
def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    """Calculate cosine similarity between two vectors using pure Python"""
    dot_product = sum(a*b for a,b in zip(vec_a, vec_b, strict=True))
    norm_a = sqrt(sum(a*a for a in vec_a))
    norm_b = sqrt(sum(b*b for b in vec_b))
    return dot_product / (norm_a * norm_b + 1e-10)  # Small epsilon to avoid division by zero

def find_semantic_match(user_input: str) -> str | None:
    """
    Check cache for semantically similar questions.
    Returns (cached_answer, similarity_score) if found, else None.
    """
    if not st.session_state["answer_cache"]:
        return None
    input_embedding = EMBEDDING_MODEL.embed_query(user_input)
    best_match = None
    highest_sim = 0.0
    
    for cached_embedding, cached_answer in st.session_state["answer_cache"].values():
        sim = cosine_similarity(input_embedding, cached_embedding)
        if sim > highest_sim:
            highest_sim = sim
            best_match = cached_answer

    return best_match if highest_sim >= SEMANTIC_SIMILARITY_THRESHOLD else None

def generate_md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def generate_md5_hash(question: str) -> str:
    """Generate MD5 hash for any input text"""
    return hashlib.md5(question.encode("utf-8")).hexdigest()

def getInitialVectorstore() -> FAISS | None:
    try:
        return FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    except:
        return FAISS.from_texts([], embedding=EMBEDDING_MODEL)

# =======================
# Session and UI Utilities
# =======================
def scroll_to_bottom() -> None:
    components.html("""<script>window.scrollTo(0, document.body.scrollHeight);</script>""", height=0)

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

def reset() -> None:
    st.session_state["answer_cache"] = TTLCache(maxsize=100, ttl=3600)
    st.session_state["cooldownBeginTimestamp"] = None
    st.session_state["in_progress"] = False
    st.session_state["messages"] = []
    st.session_state["messageTimes"] = []
    st.session_state["vectorstore"] = getInitialVectorstore()
    st.session_state["uninitialized"] = False

# =======================
# FlashRank + LangChain Chat
# =======================
def rerank_results(question: str, documents: list[Document]) -> list[Document]:
    """Rerank search results using FlashRank without comparing Document objects directly."""
    if not documents:
        return []
    
    # Create pairs for FlashRank
    pairs = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
    # Get sorted pairs from FlashRank
    results = RERANKER.rerank(RerankRequest(question, pairs))
    # Reorder documents based on sorted indices, taking top 5
    return [documents[result["id"]] for result in results[:5]]

def truncate_input(messages: list[tuple[str, str]]) -> list[tuple[str, str]]:
    combined_text = []
    for msg in reversed(messages):
        msg_length = sum(len(part) for part in msg)
        if len(combined_text) + msg_length > MAX_AI_INPUT_CHARACTERS:
            break
        combined_text.append(msg)
    combined_text.reverse()
    return combined_text

def mainPage() -> None:
    st.html("<style>body { background-color: #007BFF !important; color: white !important; }</style>")
    st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Education Abroad Chatbot</h1>")
    st.html("<p align=\"center\">This is a chatbot for answering questions about CSUSB's Education Abroad program, based on the details from its website (<a href=\"https://goabroad.csusb.edu\">goabroad.csusb.edu</a>). It cannot answer questions requiring details beyond that website.</p>")

    if st.session_state.get("uninitialized", True):
        reset()

    # Display chat history
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
    ai = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=None,
        timeout=None, # Apply timeout here for the API call itself
        max_retries=2,
        api_key=api_key,
    )

    user_input = st.chat_input("Ask about studying abroad from CSUSB...")
    if user_input and canAnswer():
        with st.chat_message("human"):
            st.markdown(user_input)
        with st.chat_message("ai"):
            placeholder = st.empty()
            full_response = ""
            if not st.session_state["in_progress"]:
                st.session_state["in_progress"] = True
                cache_key = generate_md5_hash(user_input)
                if st.session_state["answer_cache"]:
                    # 1. First try exact cache match
                    if cache_key in st.session_state["answer_cache"]:
                        _, full_response = st.session_state["answer_cache"][cache_key]
                    else:
                        # 2. Check for semantic matches
                        semantic_match = find_semantic_match(user_input)
                        if semantic_match:
                            full_response = semantic_match

                if not full_response:
                    # 3. Fallback to API call and STREAMING if no cache hits or semantic matches
                    # The timeout is now handled by the ChatGroq instance
                    initial_docs_with_scores: list[tuple[Document, float]] = st.session_state["vectorstore"].similarity_search_with_relevance_scores(user_input) if st.session_state["vectorstore"] else []
                    # Use the modified rerank_results that returns Document objects
                    ranked_docs = rerank_results(user_input, [doc[0] for doc in initial_docs_with_scores if doc[1] >= CONTEXT_RELEVANCE_THRESHOLD])
                    # st.markdown([doc[1] for doc in initial_docs_with_scores])
                    # Use the page content from the selected unique documents for context
                    # Retrieve the Document objects corresponding to the unique URLs to get their content
                    # Construct context from the content of the selected documents
                    context = " ".join([doc.page_content for doc in ranked_docs]) if ranked_docs else "None"

                    # Add context to the system prompt
                    messages = [("system", SYSTEM_PROMPT + "\n\nContext:\n" + context)] + truncate_input([(m["role"], m["content"]) for m in st.session_state["messages"][-MAX_HISTORY_TO_USE:]] + [("human", user_input)])
                    # === STREAMING IMPLEMENTATION ===
                    # Use the .stream() method provided by ChatGroq
                    try:
                        for chunk in ai.stream(messages):
                            if not chunk.content: continue
                            full_response += chunk.content # Accumulate chunks
                            # Display chunk and a typing indicator (optional, but good for UX)
                            placeholder.markdown(full_response + "▌")
                    except Exception as e:
                        # Ensure full_response is set even on other errors before displaying
                        st.error(f"Error generating response: {str(e)}") # Keep the st.error for visibility outside the placeholder if needed
                        st.session_state["in_progress"] = False

                    if full_response:
                        # --- Append References ---
                        # Append references if unique URLs were found and the AI didn't respond with the "not enough information" message
                        full_response += "\n\nReference: " + (f"[Source]({ranked_docs[0].metadata['url']})" if ranked_docs and "url" in ranked_docs[0].metadata else "*[No website pages discussed the provided query]*")

                        # Store in cache after streaming is complete, only if it came from the API
                        embedding = EMBEDDING_MODEL.embed_query(user_input)
                        # Cache the *final* response including references
                        st.session_state["answer_cache"][cache_key] = (embedding, full_response)
                placeholder.markdown(full_response)
                st.session_state["messages"] += [
                    {"role": "human", "content": user_input},
                    {"role": "ai", "content": full_response}
                ]
                st.session_state["in_progress"] = False
        scroll_to_bottom()

if __name__ == "__main__":
    mainPage()