import streamlit as st
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from langchain.schema import Document
import PyPDF2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# 🔷 Load PDF from memory (NO saving)
def load_documents_from_memory(file_bytes):
    pdf_stream = BytesIO(file_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)

    documents = []

    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": i}
                )
            )

    return documents


# 🔷 Split text into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


# 🔷 Create FAISS vector DB (in memory)
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


# 🔷 Cache entire processing pipeline
@st.cache_resource(show_spinner="Processing document...")
def process_pdf(_file_bytes):
    documents = load_documents_from_memory(_file_bytes)
    chunks = split_documents(documents)
    vector_db = create_vector_store(chunks)
    return vector_db

# 🔷 Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_answer(context, question):
    prompt = f"""
                    You are a friendly and helpful AI assistant 😊.

                    First, understand the user's intent and the nature of the document.

                    - If the input is casual or conversational, respond in a warm and engaging way.

                    - If the question is not related to the document, politely guide the user to ask questions about the document.

                    - If the question is about a person described in the document:
                    → Respond naturally as a human would.
                    → Prefer using the person's name or an appropriate first-person style when it fits the context.
                    → Avoid vague or generic references.

                    - If the question is document-related:
                    → Answer using ONLY the provided context.

                    Guidelines:
                    - Be natural, clear, and conversational.
                    - Keep answers concise and well-structured.
                    - Do NOT use outside knowledge.
                    - Do not assume information not present in the context.

                    If the answer is not found in the context, respond politely:
                    "Sorry, I couldn't find the answer in the document."

                    Context:
                    {context}

                    Question:
                    {question}
              """

    response = model.generate_content(prompt)
    return response.text