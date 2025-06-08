import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === Build vector store from .txt files in a folder ===
def build_vector_store(doc_folder="jobconnect_docs"):
    docs = []
    for filepath in Path(doc_folder).glob("*.txt"):
        loader = TextLoader(str(filepath), encoding="utf-8")
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(raw_docs)
        docs.extend(split_docs)

        print(f"Processed: {filepath.name} ({len(split_docs)} chunks)")

    openai_key = st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("jobconnect_index")
    print("Vector store saved to: jobconnect_index")

# === Load general JobConnect vector store ===
def load_vector_store():
    openai_key = st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    return FAISS.load_local("jobconnect_index", embeddings, allow_dangerous_deserialization=True)

# === Load MBTI vector store (Cloud Safe) ===
def load_mbti_vector_store():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local("chatbot/mbti_vectorstore", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Warning: Failed to load MBTI vector store: {e}")
        return None

# === (Optional) Build MBTI vector store (if needed) ===
def build_mbti_vector_store(doc_path="chatbot/mbti_guide.txt"):
    loader = TextLoader(doc_path, encoding="utf-8")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("chatbot/mbti_vectorstore")
    print("MBTI vector store saved to: chatbot/mbti_vectorstore")
