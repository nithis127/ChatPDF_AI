import streamlit as st
from dotenv import load_dotenv
import base64
import hashlib

load_dotenv()

from utils import load_pdf, split_text, create_vector_store, load_qa_chain

st.set_page_config(page_title="ChatPDF AI", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
.title {
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    font-size: 16px;
    color: #9aa0a6;
}
.stButton>button {
    border-radius: 10px;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PAGE STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "chat"

# ---------------- CACHE ----------------
@st.cache_resource
def get_vector_db(file_hash, docs):
    return create_vector_store(docs)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📂 Navigation")

    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

    if uploaded_file:

        # ✅ Read file safely
        file_bytes = uploaded_file.getvalue()

        # ✅ Create unique hash (for caching)
        file_hash = hashlib.md5(file_bytes).hexdigest()

        st.session_state.pdf_bytes = file_bytes
        st.session_state.file_hash = file_hash

        with st.spinner("⚙️ Processing document..."):
            docs = load_pdf(file_bytes)
            chunks = split_text(docs)

            if len(chunks) == 0:
                st.error("⚠️ This PDF has no readable text (maybe scanned). Try another file.")
            else:
                db = get_vector_db(file_hash, chunks)
                st.session_state.qa = load_qa_chain(db)
                st.success("✅ Ready to chat!")

    st.markdown("---")

    if st.button("💬 Chat"):
        st.session_state.page = "chat"

    if st.button("📄 Preview"):
        st.session_state.page = "pdf"

# ---------------- CHAT PAGE ----------------
if st.session_state.page == "chat":

    st.markdown('<div class="title">🤖 ChatPDF AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask questions from your PDF instantly</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🧹 Clear"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("💬 Ask something about your document...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        if "qa" not in st.session_state:
            response = "⚠️ Please upload a PDF first"
        else:
            try:
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.qa.invoke({"query": user_input})
                    response = result["result"]
            except Exception as e:
                if "429" in str(e):
                    response = "⚠️ API quota exceeded"
                elif "403" in str(e):
                    response = "⚠️ API blocked"
                else:
                    response = "⚠️ Something went wrong"

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

# ---------------- PDF PAGE ----------------
elif st.session_state.page == "pdf":

    st.markdown('<div class="title">📄 Document Preview</div>', unsafe_allow_html=True)

    if "pdf_bytes" in st.session_state:
        base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode("utf-8")

        st.markdown(
            f"""
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="600px">
            </iframe>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Upload a PDF first")