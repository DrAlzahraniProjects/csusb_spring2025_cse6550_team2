import streamlit as st
import time
import torch
import os
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from huggingface_hub import login
from langchain.prompts import PromptTemplate

# ==============================================================================================================================
# WARNING:
# Streamlit reruns the majority of this script, including global declarations,
# each time it processes a message. To keep state, store non-constant data in st.session_state["..."].
# ==============================================================================================================================

COOLDOWN_CHECK_PERIOD: float = 60.0
MAX_MESSAGES_BEFORE_COOLDOWN: int = 10
COOLDOWN_DURATION: float = 180.0

def canAnswer() -> bool:
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

    remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
    st.write(
        f"ERROR: You've reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} question"
        f"{'' if MAX_MESSAGES_BEFORE_COOLDOWN == 1 else 's'} per "
        f"{int(COOLDOWN_CHECK_PERIOD//60)} minute"
        f"{'' if COOLDOWN_CHECK_PERIOD//60 == 1 else 's'}"
        f"{' ' + str(int(COOLDOWN_CHECK_PERIOD % 60)) + ' second' + ('' if int(COOLDOWN_CHECK_PERIOD % 60) == 1 else 's') if COOLDOWN_CHECK_PERIOD % 60 else ''} "
        "because the server has limited resources. Please try again in "
        f"{int(remainingTime//60)} minute"
        f"{'' if remainingTime//60 == 1 else 's'}"
        f"{' ' + str(int(remainingTime % 60)) + ' second' + ('' if int(remainingTime % 60) == 1 else 's') if remainingTime % 60 else ''}."
    )
    return False

try:
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=huggingface_token)
except KeyError:
    st.error("❌ Environment variable `HUGGINGFACE_TOKEN` is missing! Please set it before running the app.")
except Exception:
    st.error("⚠️ An error occurred loading the Hugging Face key.")

system_prompt = """
You are an expert study abroad assistant, designed to help students with all questions 
related to studying abroad. You provide short, accurate, and helpful information about scholarships, visa 
processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide. 
You remain professional, encouraging, and optimistic at all times, ensuring students feel supported and 
motivated to pursue their dreams of studying overseas. Try to answer everything in short but understandable. Do not suggest any question to user or anything. Don't say anything that were not asked. Always write short answer and accurately. 
"""

prompt_template = PromptTemplate(
    input_variables=["system_prompt", "user_input"],
    template="""
{system_prompt}

### Current Query ###
User: {user_input}
Assistant:
"""
)

@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/kaggle/working/",
        use_auth_token=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="/kaggle/working/",
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=True
    )
    return hf_model, tokenizer

def generate_stream(
    model, tokenizer, prompt, max_new_tokens=124, temperature=0.8, top_p=0.95, do_sample=True
):
    encoded_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generation_kwargs = dict(
        **encoded_inputs,
        max_new_tokens=124,
        # temperature=temperature,
        # top_p=top_p,
        # do_sample=do_sample,
        streamer=streamer
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

    thread.join()

st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")

if "cooldownBeginTimestamp" not in st.session_state:
    st.session_state["cooldownBeginTimestamp"] = None
if "messages" not in st.session_state or not isinstance(st.session_state["messages"], list):
    st.session_state["messages"] = []
if "messageTimes" not in st.session_state or not isinstance(st.session_state["messageTimes"], list):
    st.session_state["messageTimes"] = []

hf_model, tokenizer = load_model()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is your question?")
responseStartTime, responseEndTime = 0.0, 0.0

if prompt and canAnswer():
    st.chat_message("human").markdown(prompt)
    st.session_state["messages"].append({"role": "human", "content": prompt})

    conversation_history = "\n".join(
        f"User: {msg['content']}" if msg["role"] == "human" else f"Assistant: {msg['content']}"
        for msg in st.session_state["messages"]
    )

    final_prompt = prompt_template.format(
        system_prompt=system_prompt,
        user_input=prompt
    )

    responseStartTime = time.monotonic()
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        partial_output = ""
        for token_text in generate_stream(
            hf_model, 
            tokenizer, 
            prompt=final_prompt,
            max_new_tokens=128,
            # temperature=0.8,
            # top_p=0.95,
            # do_sample=True
        ):
            partial_output = token_text
            text_placeholder.markdown(partial_output + "▌")

        # Remove the cursor "▌" at the end
        text_placeholder.markdown(partial_output)
    responseEndTime = time.monotonic()

    st.session_state["messages"].append({"role": "ai", "content": partial_output})

if responseEndTime:
    st.write(f"*(Last response took {responseEndTime - responseStartTime:.4f} seconds)*")
