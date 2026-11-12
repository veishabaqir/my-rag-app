import streamlit as st
import os
import tempfile
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="My RAG App", page_icon="📚")
st.title("📚 My RAG App - Chat With Your PDF")
st.write("Upload a PDF, then ask any question from it!")

# Sidebar - API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Paste your key saved in WhatsApp")
    st.info("Get key from aistudio.google.com/apikey")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

if uploaded_file and api_key:
    if st.button("Process PDF"):
        with st.spinner("Reading your PDF... Please wait..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Read PDF
            reader = PdfReader(tmp_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)

            # Create Embeddings + Vector Store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
            vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            st.success(f"Done! I read {len(chunks)} chunks from your PDF. Now ask questions below!")
            os.remove(tmp_path)

elif uploaded_file and not api_key:
    st.warning("Please enter your Gemini API Key in sidebar first!")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question from your PDF..."):
    if not st.session_state.vectorstore:
        st.warning("Please upload and Process PDF first!")
    elif not api_key:
        st.warning("Please enter API Key first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search relevant chunks
                docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in docs])

                llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key, temperature=0.3)
                full_prompt = f"""You are a helpful assistant. Answer ONLY from the context below.
                If answer is not in context, say 'I could not find this in your PDF'.

                Context:
                {context}

                Question: {prompt}
                Answer:"""
                response = llm.invoke(full_prompt)
                answer = response.content
                if isinstance(answer, list):
                 answer ="".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in answer])
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})