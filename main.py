from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from functions import get_pdf_text, get_text_chunks, get_vector_store


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.title("QA Bot with Gemini & RAG")



# The memory component missing in this project as to be handling multi-turn convos
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None



#File uploader
uploaded_files = st.file_uploader("Upload PDF(s)", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):

        raw_text = get_pdf_text(uploaded_files)
        st.info(f"Extracted {len(raw_text)} characters")

        chunks = get_text_chunks(raw_text)
        st.info(f"✂️ Split into {len(chunks)} chunks")

        #Create/Load vector store ONCE
        if st.session_state.vectordb is None and chunks:
            st.session_state.vectordb = get_vector_store(chunks)

        if st.session_state.vectordb is None:
            st.error("❌ Failed to create vector store. Check your PDFs contain readable text.")

    # Safe retriever setup (your other fix)
    if st.session_state.vectordb is not None:
        retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
        # ... rest of chat interface

        # Chat interface
        user_q = st.text_input("💬 Ask a question about your PDF(s):")



#auto poy mou eipe na trekso tora


        if st.button("🔍 Get Answer", type="primary") and user_q:
            with st.spinner("Generating AI answer..."):
                docs = retriever.invoke(user_q)
                if not docs:
                    st.warning("No relevant documents found.")
                else:
                    context = "\n\n".join(doc.page_content for doc in docs)

                    prompt = f"""Use ONLY the following context to answer the question. 
        If you don't know the answer, say so.

        Context: {context}

        Question: {user_q}

        Answer:"""

                    # DEBUG - SHOW EXACT ERROR
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        response = model.generate_content(prompt)
                        st.success("✅ SUCCESS!")
                        st.markdown(response.text)
                    except:
                        try:
                            model = genai.GenerativeModel('gemini-2.5-pro')
                            response = model.generate_content(prompt)
                            st.success("✅ AI Answer generated!")
                            st.markdown(response.text)
                        except:
                            st.warning("Gemini failed, showing context:")
                            st.markdown(context[:2000])

else:
    st.info("👆 Please upload PDF files to start!")
    st.caption("Supports multiple PDFs. Vector store persists across reruns.")




# Sidebar info for the understanding of user
with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown("""
    1. **Upload PDFs** → Text extraction
    2. **Chunking** → Split into 1000-char pieces  
    3. **Vector Store** → FAISS index (saved to disk)
    4. **Query** → Retrieve relevant chunks + Gemini answer
    """)
    st.caption("Uses Gemini 1.5 Pro + text-embedding-004")