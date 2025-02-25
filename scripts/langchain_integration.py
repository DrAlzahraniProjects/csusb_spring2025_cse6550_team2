import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import LLAMA_MODEL_NAME
from langchain_groq import ChatGroq

def run_qa_chain(query, segments=None):
    """
    Runs the retrieval QA chain for a given query.
    If segments are not provided, a default list is used.
    """
    # Prompt for the API key if not set.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()
        os.environ["GROQ_API_KEY"] = api_key

    # If no segments are provided, use default segments (ideally load from your JSON output)
    if segments is None:
        segments = [
            "Example segment 1 text...",
            "Example segment 2 text...",
            # ... add more segments or load them dynamically from your scraped data.
        ]
    
    # 1. Create an embeddings wrapper using HuggingFaceEmbeddings.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Build a FAISS vector store from the text segments.
    vectorstore = FAISS.from_texts(segments, embeddings)
    
    # 3. Instantiate your Groq API client using ChatGroq.
    llm = ChatGroq(
        model=LLAMA_MODEL_NAME,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
    )
    
    # 4. Set up the RetrievalQA chain using the 'stuff' chain type.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # 5. Return the answer for the given query.
    return qa_chain.invoke(query)

# For testing purposes, run this file directly:
if __name__ == "__main__":
    sample_query = "What study abroad programs does CSUSB offer?"
    answer = run_qa_chain(sample_query)
    print("Response:", answer)
