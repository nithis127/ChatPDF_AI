import streamlit as st
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer

load_dotenv()

from utils import load_pdf, split_text, create_vector_store, load_qa_chain

st.set_page_config(page_title="AI PDF Chat", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Title */
.title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 0px;
}

/* Subtitle */
.subtitle {
    font-size: 16px;
    color: #9aa0a6;
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 500;
}

/* Chat bubble spacing */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px;
}

/* Sidebar */
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
def get_vector_db(docs):
    return create_vector_store(docs)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📂 Navigation")

    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.pdf_path = "temp.pdf"

        with st.spinner("⚙️ Processing document..."):
            docs = load_pdf("temp.pdf")
            chunks = split_text(docs)

            if len(chunks) == 0:
                st.error("⚠️ This PDF has no readable text (maybe scanned). Try another file.")
            else:
                db = get_vector_db(chunks)
                st.session_state.qa = load_qa_chain(db)
                st.success("✅ Ready to chat!")

    st.markdown("---")

    if st.button("💬 Chat"):
        st.session_state.page = "chat"

    if st.button("📄 Preview"):
        st.session_state.page = "pdf"

# ---------------- CHAT PAGE ----------------
if st.session_state.page == "chat":

    # Header
    st.markdown('<div class="title">🤖 AI PDF Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask questions from your PDF instantly</div>', unsafe_allow_html=True)

    # Clear button
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🧹 Clear"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("💬 Ask something about your document...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        if "qa" not in st.session_state:
            response = "⚠️ Please upload a PDF first"
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.qa.invoke({"query": user_input})
                    response = result["result"]
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        response = "⚠️ API quota exceeded. Please wait or try later."
                    else:
                        response = "⚠️ Something went wrong. Try again."

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        with st.chat_message("assistant"):
            st.markdown(response)

# ---------------- PDF PAGE ----------------
elif st.session_state.page == "pdf":

    st.markdown('<div class="title">📄 Document Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">View your uploaded PDF</div>', unsafe_allow_html=True)

    if "pdf_path" in st.session_state:
        pdf_viewer(st.session_state.pdf_path)
    else:
        st.warning("⚠️ Upload a PDF to preview")