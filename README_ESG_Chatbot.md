# ESG Report Chatbot for IT Companies

An AI-powered ESG (Environmental, Social, Governance) chatbot that
allows users to upload ESG reports (PDFs) of IT companies and ask
intelligent, context-aware questions using Retrieval-Augmented
Generation (RAG).

------------------------------------------------------------------------

## Project Overview

This project enables users to:

-   Upload ESG reports in PDF format\
-   Automatically extract and split document content\
-   Convert text into embeddings\
-   Store embeddings in a FAISS vector database\
-   Ask ESG-related questions\
-   Get accurate answers grounded in the uploaded document

The chatbot uses **LangChain + FAISS + HuggingFace Embeddings + Groq
LLM** to deliver contextual answers.

------------------------------------------------------------------------

## Features

-  PDF ESG report upload\
-  Semantic search using vector embeddings\
-  Context-based question answering\
-  Interactive Streamlit UI\
-  Fast inference using Groq LLM\
-  Designed specifically for IT company ESG analysis

------------------------------------------------------------------------

## Tech Stack

-   Python\
-   Streamlit\
-   LangChain\
-   FAISS (Vector Database)\
-   HuggingFace Sentence Transformers\
-   Groq LLM\
-   dotenv

------------------------------------------------------------------------

## Project Structure

ESG-Chatbot/ │ ├── app.py \# Main Streamlit application ├──
requirements.txt \# Required dependencies ├── .env \# API keys (not
pushed to GitHub) └── README.md \# Project documentation

------------------------------------------------------------------------

## Installation & Setup

### 1 Clone the repository

git clone https://github.com/your-username/esg-chatbot.git cd
esg-chatbot

### 2 Create virtual environment (recommended)

python -m venv venv source venv/bin/activate \# Mac/Linux
venv`\Scripts`{=tex}`\activate      `{=tex}\# Windows

### 3 Install dependencies

pip install -r requirements.txt

### 4 Add API Key

Create a `.env` file:

GROQ_API_KEY=your_api_key_here

------------------------------------------------------------------------

##  Run the Application

streamlit run app.py

Then open the local URL shown in the terminal.

------------------------------------------------------------------------

## How It Works (Architecture)

1.  PDF is uploaded\
2.  PyPDFLoader extracts text\
3.  RecursiveCharacterTextSplitter splits text\
4.  HuggingFaceEmbeddings converts text to vectors\
5.  FAISS stores embeddings\
6.  RetrievalQA retrieves relevant chunks\
7.  Groq LLM generates final answer

------------------------------------------------------------------------

## Sample ESG Questions You Can Ask

-   What are the company's carbon emission targets?\
-   How does the company manage data privacy risks?\
-   What diversity and inclusion initiatives are mentioned?\
-   What governance policies are in place?\
-   What are Scope 1, 2, and 3 emissions reported?

------------------------------------------------------------------------

## Use Case

This project is useful for:

-   ESG Analysts\
-   Sustainability Consultants\
-   Investors evaluating IT companies\
-   Data analysts performing ESG benchmarking

------------------------------------------------------------------------

## Future Improvements

-   Multi-company ESG comparison\
-   ESG KPI dashboard visualization\
-   Chat history memory\
-   Deployment on AWS / Azure\
-   Role-based question templates (CEO / CMO view)

------------------------------------------------------------------------

## Why This Project Matters

ESG reporting is becoming mandatory for many organizations. This chatbot
simplifies ESG document analysis by enabling natural language
interaction with lengthy reports.

------------------------------------------------------------------------

## Author

**Harikaran S M**\
AI/ML & Data Science Enthusiast\
Chennai, India
