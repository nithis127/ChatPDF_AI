import streamlit as st
from dotenv import load_dotenv
import base64
import hashlib

load_dotenv()

from utils import load_pdf, split_text, create_vector_store, load_qa_chain

st.set_page_config(page_title="ChatPDF AI", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
.title { font-size: 36px; font-weight: 700; }
.subtitle { color: #9aa0a6; }
.stButton>button { border-radius: 10px; }
section[data-testid="stSidebar"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "chat"

if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------- CACHE ----------------
@st.cache_resource
def get_vector_db(file_hash, docs):
    return create_vector_store(docs)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📂 Navigation")

    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # 🔥 PROCESS ONLY IF NEW FILE
        if st.session_state.get("file_hash") != file_hash:

            st.session_state.file_hash = file_hash
            st.session_state.pdf_bytes = file_bytes
            st.session_state.processed = False

        # 🔥 PROCESS ONLY ONCE
        if not st.session_state.processed:

            with st.spinner("⚙️ Processing document..."):

                docs = load_pdf(file_bytes)
                chunks = split_text(docs)

                if len(chunks) == 0:
                    st.error("⚠️ No readable text found")
                else:
                    db = get_vector_db(file_hash, chunks)
                    st.session_state.qa = load_qa_chain(db)
                    st.session_state.processed = True

            st.success("✅ Ready to chat!")

    st.markdown("---")

    if st.button("💬 Chat"):
        st.session_state.page = "chat"

    if st.button("📄 Preview"):
        st.session_state.page = "pdf"

# ---------------- CHAT ----------------
if st.session_state.page == "chat":

    st.markdown('<div class="title">🤖 ChatPDF AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Chat with your PDF instantly</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🧹"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("💬 Ask something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        if "qa" not in st.session_state:
            response = "⚠️ Upload a PDF first"
        else:
            try:
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.qa.invoke({"query": user_input})
                    response = result["result"]
            except Exception:
                response = "⚠️ Error occurred"

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

# ---------------- PREVIEW ----------------
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