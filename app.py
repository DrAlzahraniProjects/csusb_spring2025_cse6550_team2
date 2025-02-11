from langchain_groq import ChatGroq
import os
import streamlit as st
import time

# ==============================================================================================================================
# WARNING:
# Streamlit apparently has the horrific property of RERUNNING THE MAJORITY OF THE CODE, INCLUDING GLOBAL VARIABLE DECLARATIONS,
# EVERY TIME IT REACHES THE END OF THIS FILE (currently, after each pair of human + AI messages).
# To have (non-constant) variables persist beyond that, you must instead store their values in st.session_state["..."] instead.
# ==============================================================================================================================

COOLDOWN_CHECK_PERIOD: float = 60.
MAX_MESSAGES_BEFORE_COOLDOWN: int = 10
COOLDOWN_DURATION: float = 180.
MAX_RESPONSE_TIME: float = 3.
SHOWN_API_KEY_CHARACTERS_COUNT: int = 8

SYSTEM_PROMPT: str = """
You are an expert assistant for the Study Abroad program of California State University, San Bernardino (CSUSB). You are designed to help students with all questions related to studying abroad.
You provide detailed, accurate, and helpful information about scholarships, visa processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide.
You remain professional, encouraging, and optimistic at all times, ensuring students feel supported and motivated to pursue their dreams of studying overseas.
Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, visas, or life as an international student. If a question is unrelated (e.g. politics or unrelated personal advice), politely guide the user back to study abroad topics.
- **No Negative Responses:** While you must remain truthful at all times, also avoid negative opinions, discouragement, or any response that may deter students from studying abroad.
- **Encourage and Inform:** Provide factual, detailed, and encouraging responses to all study abroad inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad, including politics, religion, or personal debates.
"""

# Returns whether or not the AI can answer the human, primarily based on the message cooldown.
# If it cannot, it writes an error message listing when the cooldown will end.
def canAnswer() -> bool:
	# Get timestamp.
	# Its value is meaningless outside of being measured against other monotonic timestamps.
	currentTimestamp = time.monotonic()
	# If we're currently in a cooldown:
	if st.session_state["cooldownBeginTimestamp"] is not None:
		# And the necessary time has passed: cancel cooldown and return true
		if currentTimestamp - st.session_state["cooldownBeginTimestamp"] >= COOLDOWN_DURATION:
			st.session_state["cooldownBeginTimestamp"] = None
			return True
	else:
		# Otherwise if we're not in a cooldown, update list of message send times
		st.session_state["messageTimes"] = st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN:]
		st.session_state["messageTimes"].append(currentTimestamp)
		# If we haven't hit the max messages, or the difference is more than cooldown period, return true
		if len(st.session_state["messageTimes"]) <= MAX_MESSAGES_BEFORE_COOLDOWN or st.session_state["messageTimes"][-1] - st.session_state["messageTimes"][-MAX_MESSAGES_BEFORE_COOLDOWN - 1] >= COOLDOWN_CHECK_PERIOD: return True
		# Otherwise start cooldown
		else: st.session_state["cooldownBeginTimestamp"] = currentTimestamp
	# Print remaining time until cooldown ends
	cooldownMinutes = int(COOLDOWN_CHECK_PERIOD//60)
	cooldownSeconds = int(COOLDOWN_CHECK_PERIOD) % 60
	remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
	remainingMinutes = int(remainingTime//60)
	remainingSeconds = int(remainingTime) % 60
	st.write(f"ERROR: You've reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} question{'' if MAX_MESSAGES_BEFORE_COOLDOWN == 1 else 's'} per {cooldownMinutes} minute{'' if cooldownMinutes == 1 else 's'}{' ' + str(cooldownSeconds) + ' second' + ('' if cooldownSeconds == 1 else 's') if cooldownSeconds else ''} because the server has limited resources. Please try again in {remainingMinutes} minute{'' if remainingMinutes == 1 else 's'}{' ' + str(remainingSeconds) + ' second' + ('' if remainingSeconds == 1 else 's') if remainingSeconds else ''}.")
	return False


def apiBox():
	if ("GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]) and "GROQ_API_KEY" in st.session_state: os.environ["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"]
	with st.container(border=True):
		halfCharactersCount = int(SHOWN_API_KEY_CHARACTERS_COUNT//2)
		newAPIkey = st.text_input("New Groq API key:", placeholder="[New Grok API key]", type="password", label_visibility="hidden")
		st.session_state["GROQ_API_KEY"] = os.environ["GROQ_API_KEY"] = newAPIkey
		st.write(f"Your current API key is {'[none provided]' if 'GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY'] else os.environ['GROQ_API_KEY'] if len(os.environ['GROQ_API_KEY']) <= SHOWN_API_KEY_CHARACTERS_COUNT else os.environ['GROQ_API_KEY'][:halfCharactersCount] + '...' + os.environ['GROQ_API_KEY'][halfCharactersCount - SHOWN_API_KEY_CHARACTERS_COUNT:]}. Click to change.")
		# st.rerun()


def mainPage() -> None:
	st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")
	apiBox()
	if "cooldownBeginTimestamp" not in st.session_state: st.session_state["cooldownBeginTimestamp"] = None
	if "messages" not in st.session_state or not isinstance(st.session_state["messages"], list): st.session_state["messages"] = []
	if "messageTimes" not in st.session_state or not isinstance(st.session_state["messageTimes"], list): st.session_state["messageTimes"] = []
	for message in st.session_state["messages"]:
		with st.chat_message(message["role"]): st.markdown(message["content"])
	ai = ChatGroq(
		model="llama-3.1-8b-instant",
		temperature=0,
		max_tokens=None,
		timeout=None,
		max_retries=2,
		# other params...
	)
	responseStartTime, responseEndTime = 0., 0.
	prompt = st.chat_input("What is your question?", disabled='GROQ_API_KEY' not in os.environ or not os.environ['GROQ_API_KEY'])
	if prompt and canAnswer():
		st.chat_message("human").markdown(prompt)
		st.session_state["messages"].append({"role": "human", "content": prompt})
		messages = [("system", SYSTEM_PROMPT)] + [(message["role"], message["content"]) for message in st.session_state["messages"]]
		# Start timing when LLM begins processing
		responseStartTime = time.monotonic()
		with st.chat_message("ai"):
			# response = st.write_stream(generatorForLLMResponse(...))
			response = ai.invoke(messages)
			# End timing when LLM being processing
			responseEndTime = time.monotonic()
			st.markdown(response.content)
			st.session_state["messages"].append({"role": "ai", "content": response.content})
	if responseEndTime:
		responseTime = responseEndTime - responseStartTime
		st.write(f"*(Last response took {':red[**' if responseTime > MAX_RESPONSE_TIME else ''}{responseTime:.4f} seconds{'**]' if responseTime > MAX_RESPONSE_TIME else ''})*")


def main():
	os.environ["GROQ_API_KEY"] = ""
	mainPage()


main()