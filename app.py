import os
import re
import time
import json
import math
import hashlib
import requests
import streamlit as st
import streamlit.components.v1 as components
from typing import Tuple, Dict, Any, Generator, Iterable
from cachetools import TTLCache
from urllib.parse import urlparse
from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import concurrent.futures as fs

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
URL_HASHES_PATH = os.path.join(".", "data", "index", "hashes.json")
URL_HASHES: dict[str, str] = {}
if os.path.exists(URL_HASHES_PATH):
    with open(URL_HASHES_PATH, "r") as f:
        try:
            URL_HASHES = json.load(f)
        except json.JSONDecodeError:
            URL_HASHES = {}

COOLDOWN_CHECK_PERIOD = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN = 10
COOLDOWN_DURATION = 180.0
ANSWER_TYPE_MAX_CHARACTERS_TO_CHECK = 30
MAX_AI_INPUT_CHARACTERS = 5000
MAX_HISTORY_TO_USE = 8
DEBUG_MODE = False
SEGMENT_SIZE = 512
SEMANTIC_SIMILARITY_THRESHOLD = 0.95
MAX_RESPONSE_TIME = None

SYSTEM_PROMPT = """
You are Beta, an expert assistant for the Education Abroad program of California State University, San Bernardino (CSUSB).
... (prompt truncated for brevity)
"""

EMBEDDING_MODEL = FastEmbedEmbeddings()
RERANKER = Ranker(max_length=4096)
INDEX_PATH = os.path.join(".", "data", "index")
os.makedirs("data", exist_ok=True)

# =======================
# Vector Search and Similarity
# =======================
def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    dot_product = sum(a*b for a,b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a*a for a in vec_a))
    norm_b = math.sqrt(sum(b*b for b in vec_b))
    return dot_product / (norm_a * norm_b + 1e-10)

def find_semantic_match(cache: TTLCache, user_input: str) -> str | None:
    input_embedding = EMBEDDING_MODEL.embed_query(user_input)
    best_match, highest_sim = None, 0.0
    for cached_embedding, cached_answer in cache.values():
        sim = cosine_similarity(input_embedding, cached_embedding)
        if sim > highest_sim:
            highest_sim = sim
            best_match = cached_answer
    return best_match if highest_sim >= SEMANTIC_SIMILARITY_THRESHOLD else None

def generate_md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def getInitialVectorstore() -> FAISS | None:
    if DEBUG_MODE: return None
    try:
        return FAISS.load_local(INDEX_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    except:
        return None

# =======================
# Session and UI Utilities
# =======================
def scroll_to_bottom() -> None:
    components.html("""<script>window.scrollTo(0, document.body.scrollHeight);</script>""", height=0)

def canAnswer() -> bool:
    current = time.monotonic()
    if st.session_state.get("cooldownBeginTimestamp") is not None:
        if current - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
            st.session_state["cooldownBeginTimestamp"] = None
            return True
    else:
        st.session_state["messageTimes"] = st.session_state.get("messageTimes", [])[-MAX_MESSAGES_BEFORE_COOLDOWN:] + [current]
        if len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or current - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD:
            return True
        st.session_state["cooldownBeginTimestamp"] = current
    return False

def reset() -> None:
    st.session_state["cooldownBeginTimestamp"] = None
    st.session_state["messageTimes"] = []
    st.session_state["messages"] = []
    st.session_state["answer_cache"] = TTLCache(maxsize=100, ttl=3600)
    st.session_state["vectorstore"] = getInitialVectorstore()
    st.session_state["uninitialized"] = False

# =======================
# FlashRank + LangChain Chat
# =======================
def rerank_results(question: str, documents: list[Document]) -> list[Document]:
    if not documents:
        return []
    pairs = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
    results = RERANKER.rerank(RerankRequest(question, pairs))
    return [documents[result["id"]] for result in results[:5]]

def truncate_input(messages: list[tuple[str, str]]) -> list[tuple[str, str]]:
    combined_text = []
    for msg in reversed(messages):
        msg_length = sum(len(part) for part in msg)
        if len(combined_text) + msg_length > MAX_AI_INPUT_CHARACTERS:
            break
        combined_text.append(msg)
    return list(reversed(combined_text))

def handle_chat_interaction(ai: ChatGroq | None, user_input: str, cache: TTLCache | None, pastMessages: list[tuple[str, str]], vectorstore: FAISS | None) -> Generator[str, None, bool]:
    cache_key = generate_md5_hash(user_input)
    if cache and cache_key in cache:
        yield cache[cache_key][1]
        return True
    match = find_semantic_match(cache, user_input) if cache else None
    if match:
        yield match
        return True

    docs = vectorstore.similarity_search(user_input) if vectorstore else []
    ranked_docs = rerank_results(user_input, docs)
    url_to_doc = {doc.metadata.get("url", ""): doc for doc in ranked_docs if doc.metadata}
    final_segments = [doc.page_content[:500] for doc in url_to_doc.values()]
    context = " ".join(final_segments) if final_segments else "None"

    messages = [("system", SYSTEM_PROMPT + "\n\nContext:\n" + context)] + [(m["role"], m["content"]) for m in pastMessages[-MAX_HISTORY_TO_USE:]] + [("human", user_input)]
    truncated = truncate_input(messages)

    if ai:
        raw_response = ""
        try:
            for chunk in ai.stream(truncated):
                if chunk.content:
                    raw_response += chunk.content
                    yield chunk.content
        except Exception as e:
            yield f"Error generating response: {str(e)}"
            return
        if url_to_doc and raw_response.strip() != "I don't have enough information to answer this question.":
            yield f"\n\nReference: [Source]({list(url_to_doc.keys())[0]})"
        if cache:
            cache[cache_key] = (EMBEDDING_MODEL.embed_query(user_input), raw_response)
        return True
    yield "[AI model not available.]"
    return False

def _tempChatWrapper(ai, user_input, cache, pastMessages, vectorstore) -> str:
    return "".join(handle_chat_interaction(ai, user_input, cache, pastMessages, vectorstore))

def mainPage() -> None:
    st.html("""
        <style>body { background-color: #007BFF !important; color: white !important; }</style>
    """)
    st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Education Abroad Chatbot</h1>")
    st.html("<p align='center'>This is a chatbot for answering questions about CSUSB's Education Abroad program: <a href='https://goabroad.csusb.edu'>goabroad.csusb.edu</a></p>")

    if st.session_state.get("uninitialized", True):
        reset()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ_API_KEY.")
        st.stop()

    ai = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=api_key)
    user_input = st.chat_input("Ask about studying abroad from CSUSB...")
    if user_input and canAnswer():
        with st.chat_message("human"):
            st.markdown(user_input)
        with st.chat_message("ai"):
            placeholder = st.empty()
            full_response = ""
            with fs.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_tempChatWrapper, ai, user_input, st.session_state["answer_cache"], st.session_state["messages"], st.session_state["vectorstore"])
                try:
                    for word in future.result(timeout=MAX_RESPONSE_TIME):
                        full_response += word
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    st.session_state["messages"] += [
                        {"role": "human", "content": user_input},
                        {"role": "ai", "content": full_response}
                    ]
                except fs.TimeoutError:
                    st.error(f"ERROR: Failed to respond within {MAX_RESPONSE_TIME} seconds.")
        scroll_to_bottom()

def main():
    mainPage()

if __name__ == "__main__":
    main()
