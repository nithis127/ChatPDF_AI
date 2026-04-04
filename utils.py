import os
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# 🔹 Load PDF file
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# 🔹 Split text into small chunks
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # size of each chunk
        chunk_overlap=50     # overlap between chunks
    )
    return splitter.split_documents(documents)


# 🔹 Create vector database (in-memory only)
def create_vector_store(docs):
    if not docs:
        raise ValueError("No text found in PDF. Please upload a valid PDF.")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    return db


# 🔹 Create QA chain using Gemini
def load_qa_chain(db):
    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",  # Gemini model
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    # Create question-answer chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa