import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Conceptual AI", layout="wide")

# -------------------- STYLE --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}
.chat-user {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 10px;
    margin: 5px 0;
}
.chat-ai {
    background-color: #38bdf8;
    padding: 12px;
    border-radius: 10px;
    margin: 5px 0;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🧠 Conceptual AI by Sanjay Sukumar")
st.caption("AI Document Assistant (Local AI - No API) 🚀")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_model()

# -------------------- FUNCTIONS --------------------
def load_pdfs(files):
    docs = []
    for file in files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append({
                    "text": text,
                    "page": i + 1,
                    "file": file.name
                })
    return docs

def split_docs(docs, chunk_size=300, overlap=100):
    chunks = []
    for doc in docs:
        text = doc["text"]
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size]
            if len(chunk.strip()) > 50:
                chunks.append({
                    "text": chunk,
                    "page": doc["page"],
                    "file": doc["file"]
                })
    return chunks

def search_top_k(query, chunks, embeddings, k=3):
    query_embedding = embed_model.encode([query])[0]

    similarities = [
        np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        for emb in embeddings
    ]

    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []
    for i in top_indices:
        if similarities[i] > 0.3:
            results.append(chunks[i])

    return results

# 🔥 IMPROVED AI ANSWER
def generate_answer(query, contexts):
    combined_context = "\n\n".join([c["text"] for c in contexts])[:800]

    prompt = f"""
You are a professional AI assistant.

STRICT RULES:
- Answer ONLY from the given context
- DO NOT add extra information
- Keep answer SHORT and PRECISE
- Use bullet points when possible
- If answer not found, say "Not found in document"

Context:
{combined_context}

Question:
{query}

Give a clean structured answer:
"""

    try:
        response = ollama.chat(
            model='phi',
            messages=[{"role": "user", "content": prompt}]
        )

        return response['message']['content']

    except:
        return "⚠️ AI error. Make sure Ollama is running."

# -------------------- STATE --------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------- UI --------------------
files = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)

if files:
    docs = load_pdfs(files)
    chunks = split_docs(docs)

    embeddings = embed_model.encode([c["text"] for c in chunks])

    st.success(f"✅ {len(files)} file(s) processed | {len(chunks)} chunks ready")

    query = st.chat_input("Ask anything from your documents...")

    if query:
        top_chunks = search_top_k(query, chunks, embeddings)

        if not top_chunks:
            answer = "❌ Not found in document."
            st.session_state.chat.append(("user", query))
            st.session_state.chat.append(("ai", answer, []))
        else:
            answer = generate_answer(query, top_chunks)
            st.session_state.chat.append(("user", query))
            st.session_state.chat.append(("ai", answer, top_chunks))

# -------------------- CHAT DISPLAY --------------------
for item in st.session_state.chat:
    if item[0] == "user":
        st.markdown(f"<div class='chat-user'>👤 {item[1]}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>🤖 {item[1]}</div>", unsafe_allow_html=True)

        for s in item[2]:
            st.caption(f"📄 Source: {s['file']} | Page {s['page']}")