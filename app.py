import streamlit as st
from dotenv import load_dotenv
import os
import hashlib
import time

from rag_pipeline import process_pdf, generate_answer

# 🔷 Load env
load_dotenv()

# 🔷 Page config
st.set_page_config(
    page_title="ChatPDF AI",
    page_icon="💬",
    layout="wide"
)

st.title("💬 ChatPDF AI")
st.caption("Ask questions about your PDF instantly using AI 🚀")

# 🔷 Session state init
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# 🔷 Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    # 🔷 Clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    # 🔷 Auto process PDF
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()

        # 🔥 FIX: Use hash instead of raw bytes
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.file_hash != file_hash:
            st.session_state.file_hash = file_hash

            with st.spinner("📄 Processing document..."):
                try:
                    st.session_state.vector_db = process_pdf(file_bytes)
                    st.success("📄 PDF loaded successfully")
                except Exception as e:
                    st.session_state.vector_db = None  # 🔥 safety reset
                    st.error(f"❌ Error processing file: {str(e)}")

    # 🔷 Status
    if st.session_state.vector_db:
        st.success("✅ You can start asking questions.")

    st.warning("⚠️ Free-tier limit is low (~20 requests/day). Use carefully.")
    st.markdown("---")
    st.caption("⚡ Built with Gemini + RAG + FAISS | ChatPDF AI")


# 🔷 API key check
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 API key not found. Please check your .env file.")
    st.stop()


# 🔷 Empty state (FIXED)
if not st.session_state.chat_history and st.session_state.vector_db is None:
    st.info("👋 Upload a PDF and start asking questions!")


# 🔷 Display chat
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)


# 🔷 Cached response (reduce API calls)
@st.cache_data(show_spinner=False)
def cached_answer(context, query):
    return generate_answer(context, query)


# 🔷 Chat input
query = st.chat_input("Ask anything about your PDF...")

if query and query.strip():
    st.session_state.chat_history.append(("user", query))

    with st.chat_message("user"):
        st.write(query)

    # 🔷 Generate response
    if st.session_state.vector_db is None:
        response = "⚠️ Please upload a document first."
    else:
        try:
            with st.spinner("🤔 Thinking..."):

                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)

                if not docs:
                    response = "⚠️ I couldn't find relevant information in the document for your question."
                else:
                    context = "\n".join([doc.page_content for doc in docs])
                    context = context[:4000]

                    # 🔥 Use cached answer
                    response = cached_answer(context, query)

        except Exception as e:
            error_msg = str(e)

            # 🔷 Quota (daily limit)
            if "429" in error_msg or "quota" in error_msg.lower():
                response = "⚠️ Daily limit reached. Please try again after some time."

            # 🔷 Network issues
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                response = "🌐 Network issue. Please check your internet connection."

            # 🔷 Fallback
            else:
                response = "❌ Something went wrong. Please try again."

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat_history.append(("assistant", response))