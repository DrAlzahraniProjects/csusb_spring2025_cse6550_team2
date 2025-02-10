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

# Returns whether or not the AI can answer the human, primarily based on the message cooldown.
# If it cannot, it writes an error message listing when the cooldown will end.
def canAnswer() -> bool:
	# Get timestamp.
	# Its value is meaningless outside of being subtractable against other monotonic times.
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
	remainingTime = COOLDOWN_DURATION + st.session_state["cooldownBeginTimestamp"] - currentTimestamp
	st.write(f"ERROR: You've reached the limit of {MAX_MESSAGES_BEFORE_COOLDOWN} question{'' if MAX_MESSAGES_BEFORE_COOLDOWN == 1 else 's'} per {int(COOLDOWN_CHECK_PERIOD//60)} minute{'' if COOLDOWN_CHECK_PERIOD//60 == 1 else 's'}{' ' + str(COOLDOWN_CHECK_PERIOD % 60) + ' second' + ('' if COOLDOWN_CHECK_PERIOD % 60 == 1 else 's') if COOLDOWN_CHECK_PERIOD % 60 else ''} because the server has limited resources. Please try again in {int(remainingTime//60)} minute{'' if remainingTime//60 == 1 else 's'}{' ' + str(remainingTime % 60) + ' second' + ('' if remainingTime % 60 == 1 else 's') if remainingTime % 60 else ''}.")
	return False



st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")

if "messages" not in st.session_state or not isinstance(st.session_state["messages"], list): st.session_state["messages"] = []
if "cooldownBeginTimestamp" not in st.session_state: st.session_state["cooldownBeginTimestamp"] = None
if "messageTimes" not in st.session_state: st.session_state["messageTimes"] = []

for message in st.session_state["messages"]:
	with st.chat_message(message["role"]): st.markdown(message["content"])

responseStartTime, responseEndTime = 0., 0.
prompt = st.chat_input("What is your question?")
if prompt and canAnswer():
	st.chat_message("human").markdown(prompt)
	st.session_state["messages"].append({"role": "human", "content": prompt})
	# Start timing when LLM begins processing
	responseStartTime = time.monotonic()
	with st.chat_message("ai"):
		# response = st.write_stream(generatorForLLMResponse(...))
		response = "[LLM response here]"
		# End timing when LLM being processing
		responseEndTime = time.monotonic()
		st.markdown(response)
		st.session_state["messages"].append({"role": "ai", "content": response})

if responseEndTime: st.write(f"*(Last response took {responseEndTime - responseStartTime:.4f} seconds)*")