import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

system_prompt = """
You are an expert study abroad assistant, designed to help students with all questions 
related to studying abroad. You provide detailed, accurate, and helpful information about scholarships, visa 
processes, university applications, living abroad, cultural adaptation, and academic opportunities worldwide. 
You remain professional, encouraging, and optimistic at all times, ensuring students feel supported and 
motivated to pursue their dreams of studying overseas.

Rules & Restrictions:
- **Stay on Topic:** Only respond to questions related to studying abroad, scholarships, university admissions, 
  visas, and life as an international student.
- **No Negative Responses:** Avoid negative opinions, discouragement, or any response that may deter students 
  from studying abroad.
- **Reject Off-Topic Questions:** If a question is unrelated (e.g., politics, unrelated personal advice), 
  politely guide the user back to study abroad topics.
- **Encourage and Inform:** Provide factual, detailed, and encouraging responses to all study abroad inquiries.
- **No Controversial Discussions:** Do not engage in topics outside of studying abroad, including politics, 
  religion, or personal debates.
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

@st.cache_resource(show_spinner=False)
def load_model_and_pipeline():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    st.info(f"Loading the HF model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        device=device
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    st.success("Model loaded successfully.")
    return llm

@st.cache_resource(show_spinner=False)
def create_chain():
    llm = load_model_and_pipeline()
    return LLMChain(llm=llm, prompt=prompt_template, verbose=False)

#  Initialize conversation history in session state 
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def format_history(history):
    """Formats the conversation history into a single string for the prompt."""
    return "\n".join(
        f"User: {msg['user']}\nAssistant: {msg['assistant']}"
        for msg in history
    )

st.title("Study Abroad Chat Assistant")

st.markdown(
    """
This assistant is designed to help you with all questions related to studying abroad.
Enter your question below and click **Send** to get started.
"""
)

user_input = st.text_input("Enter your question:")

if st.button("Send") and user_input:
    # Display a spinner while generating a response
    with st.spinner("Generating response..."):
        conversation_str = format_history(st.session_state.conversation_history)
        chain = create_chain()
        response = chain.run(
            system_prompt=system_prompt,
            conversation_history=conversation_str,
            user_input=user_input
        )
    st.session_state.conversation_history.append({
        "user": user_input,
        "assistant": response.strip()
    })

if st.session_state.conversation_history:
    st.markdown("## Conversation History")
    for chat in st.session_state.conversation_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['assistant']}")
        st.markdown("---")
