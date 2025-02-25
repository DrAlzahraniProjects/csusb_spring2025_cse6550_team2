import streamlit as st
import random
import time

# Corpus with your answerable and unanswerable questions
answerable_questions = {
    "Does CSUSB offer study abroad programs?": "Yes, CSUSB offers a wide variety of study abroad programs across multiple countries. You can find more details on the CSUSB Study Abroad website.",
    "Can I apply for a study abroad program at CSUSB?": "Yes, you can apply for study abroad programs through the Center for International Studies & Programs at CSUSB.",
    "Is Toronto a good place for students to live while studying abroad?": "Toronto is a vibrant, multicultural city with great opportunities for students. Many study abroad programs offer opportunities to study there.",
    "Do I need a visa to study at the University of Seoul?": "Yes, if you are planning to study at the University of Seoul, you will typically need a student visa. Make sure to check the latest visa requirements with the university.",
    "Can I study in South Korea or Taiwan if I only know English?": "Yes, many programs in South Korea and Taiwan offer courses in English, so you can study there even if you don't speak the local language."
}

# Unanswerable questions (simulated for this task)
unanswerable_questions = [
    "Is there a set date for the Study Abroad 101 information sessions?",
    "Is the application deadline for Concordia University's summer semester available here?",
    "Does the chatbot provide a full list of CSUSB-approved direct enrollment universities?",
    "Does the chatbot list all available study abroad scholarships?",
    "Is the internal deadline for the Fulbright Scholarship application set by CSUSB available here?"
]

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display the header
st.html("<h1 style='text-align:center; font-size:48px'>CSUSB Travel Abroad Chatbot</h1>")

# Function to simulate the dialogue between Alpha and Beta
def ai_dialogue():
    all_questions = list(answerable_questions.keys()) + unanswerable_questions
    random.shuffle(all_questions)  # Shuffle the questions to simulate randomness

    # Allow Alpha to ask 10 questions
    for i in range(10):
        question = all_questions[i]
        
        # Alpha asks the question
        st.chat_message("human").markdown(f"**Alpha**: {question}")
        st.session_state["messages"].append({"role": "human", "content": question})
        
        # Beta responds based on whether the question is in the answerable set or not
        if question in answerable_questions:
            response = answerable_questions[question]
        else:
            response = "Sorry, I can't answer that."
        
        # Beta's response
        st.chat_message("ai").markdown(f"**Beta**: {response}")
        st.session_state["messages"].append({"role": "ai", "content": response})

        # Simulate a delay between questions
        time.sleep(2)

# Run the AI dialogue automatically when the page is loaded
if st.button("Start AI Dialogue"):
    ai_dialogue()

# Display the conversation so far
if st.session_state["messages"]:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
