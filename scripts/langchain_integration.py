import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import LLAMA_MODEL_NAME
from langchain_groq import ChatGroq

# Prompt for the API key if not already set in the environment.
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    api_key = input("Enter your Groq API key: ").strip()
    os.environ["GROQ_API_KEY"] = api_key

# Define your text segments. In practice, these might be loaded from a JSON file.
segments = [
    "Example segment 1 text...",
    "Example segment 2 text...",
    # ... add more segments or load them dynamically
]

# 1. Create an embeddings wrapper using HuggingFaceEmbeddings.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Build a FAISS vector store from your text segments.
vectorstore = FAISS.from_texts(segments, embeddings)

# 3. Instantiate your Groq API client using ChatGroq with the instant model identifier.
llm = ChatGroq(
    model=LLAMA_MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
)

# 4. Set up the RetrievalQA chain with the FAISS vector store as the retriever,
#    using the recommended factory method.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" is a supported chain type; adjust as needed.
    retriever=vectorstore.as_retriever()
)

# 5. Test the integration with a sample query using the recommended invoke() method.
query = "What study abroad programs does CSUSB offer?"

try:
    response = qa_chain.invoke(query)
    print("Response:", response)
except Exception as e:
    print("Error calling Groq API:", e)
