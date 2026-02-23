import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="ESG Multi-Report Chatbot", layout="wide")

# -----------------------
# LOAD ENV
# -----------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------
# CUSTOM CSS (ChatGPT Style)
# -----------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 20px;
}
.chat-container {
    max-height: 70vh;
    overflow-y: auto;
    padding: 10px;
}
.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.bot-msg {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# TITLE
# -----------------------
st.markdown('<div class="main-title"> ESG Multi-Report Intelligence Assistant</div>', unsafe_allow_html=True)

# -----------------------
# SIDEBAR – FILE UPLOAD
# -----------------------
with st.sidebar:
    st.header(" Upload ESG Reports")
    uploaded_files = st.file_uploader(
        "Upload one or more ESG PDF reports",
        type="pdf",
        accept_multiple_files=True
    )

# -----------------------
# SESSION STATE INIT
# -----------------------
if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# PROCESS FILES
# -----------------------
if uploaded_files:
    with st.spinner("Processing ESG reports... Please wait "):
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        for uploaded_file in uploaded_files:
            
            if uploaded_file.name not in st.session_state.vectorstores:
                
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyPDFLoader(uploaded_file.name)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                docs = splitter.split_documents(documents)

                vectorstore = FAISS.from_documents(docs, embeddings)

                st.session_state.vectorstores[uploaded_file.name] = vectorstore

    st.success("All reports processed successfully ")

# -----------------------
# LLM
# -----------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

# -----------------------
# DISPLAY CHAT HISTORY
# -----------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="user-msg"> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"> {message}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# INPUT BAR (Footer Style)
# -----------------------
user_input = st.chat_input("Ask a question about the uploaded ESG reports...")

# -----------------------
# HANDLE QUESTION
# -----------------------
if user_input and st.session_state.vectorstores:

    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Analyzing ESG reports..."):

        final_answer = ""

        for filename, vectorstore in st.session_state.vectorstores.items():

            docs = vectorstore.similarity_search(user_input, k=4)

            context = "\n\n".join([doc.page_content for doc in docs])

            page_numbers = sorted(
                list(set([doc.metadata.get("page", 0) for doc in docs]))
            )
            page_numbers = [p + 1 for p in page_numbers]

            prompt = f"""
            You are a professional ESG analyst.

            Provide a clear, structured, and concise summary based only on the context.

            Context:
            {context}

            Question:
            {user_input}
            """

            response = llm.invoke(prompt)

            final_answer += f"""
### {filename}

{response.content}

 Reference Pages: {page_numbers}

---
"""

    st.session_state.chat_history.append(("bot", final_answer))
    st.rerun()













