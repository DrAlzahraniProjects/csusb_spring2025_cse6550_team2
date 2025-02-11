import streamlit as st
import time
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from huggingface_hub import login

# ==============================================================================================================================
# WARNING:
# Streamlit apparently has the horrific property of RERUNNING THE MAJORITY OF THE CODE, INCLUDING GLOBAL VARIABLE DECLARATIONS,
# EVERY TIME IT REACHES THE END OF THIS FILE (currently, after each pair of human + AI messages).
# To have (non-constant) variables persist beyond that, you must instead store their values in st.session_state["..."] instead.
# ==============================================================================================================================

COOLDOWN_CHECK_PERIOD: float = 60.
MAX_MESSAGES_BEFORE_COOLDOWN: int = 10
COOLDOWN_DURATION: float = 180.

# Returns whether or not the AI can answer the human, primarily based on the message cooldown.
# If it cannot, it writes an error message listing when the cooldown will end.
def canAnswer() -> bool:
	currentTimestamp = time.monotonic()
	if st.session_state["cooldownBeginTimestamp"] is not None:
		if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
			st.session_state["cooldownBeginTimestamp"] = None
			return True
	else:
		st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:]
		st.session_state["messageTimes"].append(currentTimestamp)
		if len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or st.session_state["messageTimes"][-1] - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD:
			return True
		else:
			st.session_state["cooldownBeginTimestamp"] = currentTimestamp
	remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
	st.write(f"ERROR: You've reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} question{'' if MAX_MESSAGES_BEFORE_COOLDOWN == 1 else 's'} per {int(COOLDOWN_CHECK_PERIOD//60)} minute{'' if COOLDOWN_CHECK_PERIOD//60 == 1 else 's'}{' ' + str(COOLDOWN_CHECK_PERIOD % 60) + ' second' + ('' if COOLDOWN_CHECK_PERIOD % 60 == 1 else 's') if COOLDOWN_CHECK_PERIOD % 60 else ''} because the server has limited resources. Please try again in {int(remainingTime//60)} minute{'' if remainingTime//60 == 1 else 's'}{' ' + str(remainingTime % 60) + ' second' + ('' if remainingTime % 60 == 1 else 's') if remainingTime % 60 else ''}.")
	return False
try:
    huggingface_token = os.environ["HUGGINGFACE_TOKEN"]
    login(token=huggingface_token)
except KeyError:
    st.error("❌ Environment variable `HUGGINGFACE_TOKEN` is missing! Please set it before running the app.")
except Exception as e:
    st.error(f"⚠️ An error occurred loading huggingface key from environment")
system_prompt = """
You are an expert study abroad assistant, designed to help students with all questions 
related to studying abroad. You provide detailed, accurate, and helpful information about scholarships, visa 
processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide. 
You remain professional, encouraging, and optimistic at all times, ensuring students feel supported and 
motivated to pursue their dreams of studying overseas.
"""
prompt_template = PromptTemplate(
	input_variables=["system_prompt", "conversation_history", "user_input"],
	template="""
{system_prompt}
### Conversation History ###
{conversation_history}
### Current Query ###
User: {user_input}
Assistant:
"""
)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
@st.cache_resource(show_spinner=False)
def load_model_and_pipeline():
	model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	bnb_config = BitsAndBytesConfig(
		load_in_8bit=True,
		bnb_8bit_compute_dtype=torch.bfloat16, # if you use TPU x2 then keep bfloat16 else use float16

	)
	tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/kaggle/working/", use_auth_token=True)
	hf_model = AutoModelForCausalLM.from_pretrained(
		model_name,
		cache_dir="/kaggle/working/",
		quantization_config=bnb_config, # Remove this if you want to load 16bit model instead of 4 bit
		device_map="auto",
		use_auth_token=True
	)
	pipe = pipeline(
		"text-generation",
		model=hf_model, 
		tokenizer=tokenizer,
		device_map="auto",
		torch_dtype=torch.bfloat16    # if you use TPU x2 then keep bfloat16 else use float16
	)
	llm = HuggingFacePipeline(pipeline=pipe)
	return llm
@st.cache_resource(show_spinner=False)
def create_chain():
	llm = load_model_and_pipeline()
	return LLMChain(llm=llm, prompt=prompt_template, verbose=False)

st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")
if "cooldownBeginTimestamp" not in st.session_state: st.session_state["cooldownBeginTimestamp"] = None
if "messages" not in st.session_state or not isinstance(st.session_state["messages"], list): st.session_state["messages"] = []
if "messageTimes" not in st.session_state or not isinstance(st.session_state["messageTimes"], list): st.session_state["messageTimes"] = []
for message in st.session_state["messages"]:
	with st.chat_message(message["role"]): st.markdown(message["content"])

responseStartTime, responseEndTime = 0., 0.
prompt = st.chat_input("What is your question?")
if prompt and canAnswer():
	st.chat_message("human").markdown(prompt)
	st.session_state["messages"].append({"role": "human", "content": prompt})
	conversation_history = "\n".join(
		f"User: {msg['content']}" if msg["role"] == "human" else f"Assistant: {msg['content']}"
		for msg in st.session_state["messages"]
	)
	# Start timing when LLM begins processing
	responseStartTime = time.monotonic()
	with st.chat_message("ai"):
		response = create_chain().run(system_prompt=system_prompt, conversation_history=conversation_history, user_input=prompt)
		# End timing when LLM being processing
		responseEndTime = time.monotonic()
		st.markdown(response)
	st.session_state["messages"].append({"role": "ai", "content": response})
if responseEndTime: st.write(f"*(Last response took {responseEndTime - responseStartTime:.4f} seconds)*")
