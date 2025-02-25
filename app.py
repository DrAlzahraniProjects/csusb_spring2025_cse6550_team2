import streamlit as st
import os
import pandas as pd
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Initialize API key
apik = os.getenv("GROQ_API_KEY")
if not apik:
    st.error("Error: Please set your GROQ_API_Key variable.")
    st.stop()
    
# Initialize chat model
chat = init_chat_model("llama3-8b-8192", model_provider="groq")
messages = [SystemMessage(content="You are an AI assistant that will help.")]

# Define model-based prompt for communication
def get_model_based_prompt(user_query):
    return f"""
    You are an AI assistant with general knowledge. Please respond to the user's query based on your pre-trained knowledge. Do not refer to any uploaded documents or extracted text unless explicitly instructed.

    User Query: {user_query}

    Please provide the most relevant and accurate response based on your training.
    """

# Initialize confusion matrix
if 'conf_matrix' not in st.session_state:
    st.session_state.conf_matrix = np.array([[0, 0], [0, 0]])

# Display metrics in the sidebar
def display_metrics():
    TP = st.session_state.conf_matrix[0, 0]  # True Positive
    TN = st.session_state.conf_matrix[1, 1]  # True Negative
    FP = st.session_state.conf_matrix[0, 1]  # False Positive
    FN = st.session_state.conf_matrix[1, 0]  # False Negative

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Specificity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Custom CSS for sky-blue sidebar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #87CEEB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display metrics in the sidebar
    st.sidebar.write("### Metrics")
    st.sidebar.write(f"**Accuracy:** {accuracy:.2f}")
    st.sidebar.write(f"**Precision:** {precision:.2f}")
    st.sidebar.write(f"**Recall:** {recall:.2f}")
    st.sidebar.write(f"**Specificity:** {specificity:.2f}")
    st.sidebar.write(f"**F1-Score:** {f1_score:.2f}")

    # Display Confusion Matrix
    st.sidebar.write("### Confusion Matrix")
    conf_df = pd.DataFrame(
        st.session_state.conf_matrix,
        index=["Actual +", "Actual -"],
        columns=["Predicted +", "Predicted -"]
    )
    st.sidebar.table(conf_df)

# AI Podcast Conversation (Placeholder function)
def start_ai_podcast():
    questions = [
        "Does CSUSB offer study abroad programs?",
        "Is there a set date for the Study Abroad 101 information sessions?",
        "Can I apply for a study abroad program at CSUSB?",
        "Is the application deadline for Concordia University's summer semester available here?",
        "Is Toronto a good place for students to live while studying abroad?",
        "Does the chatbot provide a full list of CSUSB-approved direct enrollment universities?",
        "Do I need a visa to study at the University of Seoul?",
        "Does the chatbot list all available study abroad scholarships?",
        "Can I study in South Korea or Taiwan if I only know English?",
        "Is the internal deadline for the Fulbright Scholarship application set by CSUSB available here?"
    ]   

    for i, question in enumerate(questions):
        # AI Response generation
        messages.append(HumanMessage(content=question))
        response = chat.invoke(messages)
        ai_response = response.content if i % 2 == 0 else "I do not know!"

        # Generate a concise summary
        ai_response_summary = " ".join(ai_response.splitlines()[:5])  # First 5 lines

        ai_message = AIMessage(content=ai_response_summary)
        messages.append(ai_message)

        # Display conversation in a structured format
        with st.container():
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {ai_response_summary}")
            st.markdown("---")  # UI separator for clarity

        # Update confusion matrix based on correctness
        if "I do not know!" in ai_response:
            st.session_state.conf_matrix[1, 0] += 1  # False Negative
        else:
            st.session_state.conf_matrix[0, 0] += 1  # True Positive

# Header Section
col1, col2 = st.columns([1, 3])
with col1:
    st.image(r"logo/csusb_logo.png", width=100)
with col2:
    st.markdown("<h1 style='text-align: center;'>CSUSB Study Abroad Agent</h1>", unsafe_allow_html=True)

# Chatbot Input for Model-based Communication
user_input = st.text_input("Ask the Chatbot a Question", key="chat_input")

if st.button("Submit", key="submit_button"):
    prompt = get_model_based_prompt(user_input)
    response = chat.invoke([SystemMessage(content=prompt)])
    ai_response = response.content if response else "I do not know!"
    
    st.write(f"**User:** {user_input}")
    st.write(f"**AI Response:** {ai_response}")

    # Update confusion matrix
    if "I do not know!" in ai_response:
        st.session_state.conf_matrix[1, 0] += 1  # False Negative
    else:
        st.session_state.conf_matrix[0, 0] += 1  # True Positive

# Podcast Start Button
if st.button("Start AI Podcast", key="start_podcast_button"):
    start_ai_podcast()

# Always display metrics in the sidebar
display_metrics()
