import streamlit as st
from dotenv import load_dotenv
import hashlib

load_dotenv()

from utils import load_pdf, split_text, create_vector_store, load_qa_chain

st.set_page_config(page_title="AI PDF Chat", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
.title { font-size: 36px; font-weight: 700; }
.subtitle { font-size: 16px; color: #9aa0a6; }
.stButton>button { border-radius: 10px; padding: 8px 16px; }
[data-testid="stChatMessage"] { border-radius: 12px; padding: 10px; }
section[data-testid="stSidebar"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# ---------------- CACHE ----------------
@st.cache_resource
def get_vector_db(chunks):
    return create_vector_store(chunks)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📂 Upload PDF")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.file_hash != current_hash:
            st.session_state.file_hash = current_hash
            st.session_state.processed = False

        if not st.session_state.processed:
            with st.spinner("⚙️ Processing document..."):
                docs = load_pdf(file_bytes)
                chunks = split_text(docs)

                if len(chunks) == 0:
                    st.error("⚠️ No readable text found in PDF")
                else:
                    db = get_vector_db(chunks)
                    st.session_state.qa = load_qa_chain(db)
                    st.session_state.processed = True

            st.success("✅ Ready to chat!")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.qa = None
        st.session_state.processed = False
        st.rerun()

# ---------------- MAIN ----------------
st.markdown('<div class="title">🤖 AI PDF Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions from your PDF instantly</div>', unsafe_allow_html=True)

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("💬 Ask something about your document...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if "qa" not in st.session_state:
        response = "⚠️ Please upload a PDF first"

    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # 🧠 Chat history context
                history = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
                )

                query = f"""
                Chat History:
                {history}

                Question:
                {user_input}
                """

                result = st.session_state.qa.invoke({"query": query})

                response = result["result"]

                # 📄 Source display
                sources = result.get("source_documents", [])
                if sources:
                    response += "\n\n📄 **Sources:**\n"
                    for doc in sources:
                        page = doc.metadata.get("page", "N/A")
                        response += f"- Page {page + 1}\n"

            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    response = "⚠️ API quota exceeded. Try later."
                else:
                    response = "⚠️ Something went wrong."

    st.session_state.messages.append({"role": "assistant", "content": response})

    # ---------------- STREAMING EFFECT ----------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for word in response.split():
            full_response += word + " "
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)