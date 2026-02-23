import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="ESG Report Chatbot", layout="wide")

st.markdown("<h1 style='text-align: center;'> ESG Chatbot</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload ESG PDF Reports",
    type="pdf",
    accept_multiple_files=True
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:

    with st.spinner("Processing ESG Reports... Please wait !!"):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstores = {}

        for uploaded_file in uploaded_files:

            file_path = f"temp_{uploaded_file.name}"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(docs, embeddings)

            vectorstores[uploaded_file.name] = vectorstore

    st.success("Reports processed successfully ")


    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    question = st.text_input("Ask a question about the ESG reports:")

    if question:

        st.session_state.chat_history.append(("User", question))

        with st.spinner("Analyzing reports..."):

            final_answer = ""

            for filename, vectorstore in vectorstores.items():

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True
                )

                response = qa(question)

                answer = response["result"]

               
                source_docs = response["source_documents"]
                page_numbers = sorted(
                    list(set([doc.metadata.get("page", 0) for doc in source_docs]))
                )

                
                page_numbers = [p + 1 for p in page_numbers]

                final_answer += f"""
 **{filename}**

{answer}

 Reference Pages: {page_numbers}

---
"""

        st.session_state.chat_history.append(("Assistant", final_answer))

        
        with open("chat_history.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n===== {datetime.now()} =====\n")
            f.write(f"Question: {question}\n")
            f.write(f"Answer:\n{final_answer}\n")

    
    for role, message in st.session_state.chat_history:
        if role == "User":
            st.markdown(f"### {role}")
            st.write(message)
        else:
            st.markdown(f"### {role}")
            st.write(message)
