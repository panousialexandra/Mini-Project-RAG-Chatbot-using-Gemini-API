from io import BytesIO
import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

#Function where you give one/some pdfs and get the text from the pdfs
@st.cache_data
def get_pdf_text(pdf_docs):
    """""Extract text from uploaded PDF file."""
    text = ""
    for pdf_file in pdf_docs:
        # Reset file pointer - CRITICAL for Streamlit
        pdf_file.seek(0)
        # Convert Streamlit UploadedFile to PyPDF2 format
        pdf_reader = PdfReader(BytesIO(pdf_file.read()))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text: #skip empty pages
                text += page_text + "\n"
    return text.strip() #remove loading/trailing whitespace



#Function where you give the text gotten from the pdfs and turn it into chunks (text splitting)
def get_text_chunks(raw_text):
    """""Split raw text into overlapping chunks."""

    if not raw_text or len(raw_text.strip()) < 50:  # Skip if too short
        st.warning("Extracted text too short. Check PDF content.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    #Convert raw text to Document objects
    docs = [Document(page_content=raw_text)]
    chunks= text_splitter.split_documents(docs)

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    st.info(f"Created {len(chunks)} non-empty chunks")

    return chunks



#Function where you give the split text (chunks) and get embedding (vectors and indexes)
def get_vector_store(doc_chunks, index_path="faiss_index"):

    """""Create or load FAISS vector store with embeddings."""

    if not doc_chunks:
        st.error("No valid chunks to embed. Check PDF extraction.")
        return None

    embeddings= GoogleGenerativeAIEmbeddings(model= "models/text-embedding-004", google_api_key = os.getenv("GOOGLE_API_KEY"))

    #Checking if the index already exists
    if os.path.exists(index_path):
        try:
            vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            st.info("Loaded existing vector store")
        except Exception as e:
            st.warning(f"Failed to load index: {e}. Creating a new one.")
            vectordb = FAISS.from_documents(doc_chunks,embeddings)
            vectordb.save_local(index_path)
    else:
        vectordb = FAISS.from_documents(doc_chunks, embeddings)
        vectordb.save_local(index_path)
        st.success("Created new vector store.")

    return vectordb


