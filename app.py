import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ========== Configure Gemini ==========
genai.configure(api_key="AIzaSyBeoYwJuJSaOGyWbNwzgoGl8rb2OtctSN8")

# ========== Load Embedding Model ==========
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ========== Extract Text from PDF ==========
def extract_text(file):
    try:
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

# ========== Compute Similarity ==========
def get_most_similar_docs(query, texts, top_k=3):
    query_vec = model.encode([query])
    text_vecs = model.encode(texts)
    sims = cosine_similarity(query_vec, text_vecs)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [texts[i] for i in top_indices]

# ========== Ask Gemini ==========
def ask_gemini(prompt):
    try:
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

# ========== Citation ==========
def get_citation(text, query):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if query.lower() in line.lower():
            return f"Page {i//25 + 1}, Line {i%25 + 1}"
    return "Not Found"

# ========== Streamlit App ==========
st.set_page_config(page_title="EduMentor – Gemini Chatbot", layout="wide")
st.title("📘 EduMentor – Policy Research Chatbot (Gemini-Powered)")

uploaded_files = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)
query = st.text_input("🔍 Ask your question:", placeholder="Example: What is the National Education Policy?")
submit = st.button("✍️ Get Answer")

if submit and query:
    with st.spinner("Processing documents..."):
        doc_texts = []
        doc_info = []

        for i, file in enumerate(uploaded_files):
            doc_id = f"DOC{i+1:03d}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            text = extract_text(tmp_path)
            doc_texts.append(text)
            doc_info.append((doc_id, text, file.name))

        matched_texts = get_most_similar_docs(query, doc_texts)

        matched_docs_display = []
        table_data = []

        for doc_id, text, name in doc_info:
            if text in matched_texts:
                citation = get_citation(text, query)
                snippet = re.findall(rf"(.{{0,100}}{re.escape(query)}.{{0,200}})", text, flags=re.IGNORECASE)
                answer_text = snippet[0].strip() if snippet else "Relevant info found."
                matched_docs_display.append(f"{doc_id}: {answer_text}")
                table_data.append({
                    "Document ID": doc_id,
                    "Extracted Answer": answer_text,
                    "Citation": citation,
                    "Source File": name
                })

        if matched_docs_display:
            joined = "\n".join(matched_docs_display)
            prompt = f"Answer the following question from the document snippets:\n\n{joined}\n\nQ: {query}"
            answer = ask_gemini(prompt)

            theme_prompt = f"Identify clear themes from the following snippets. Format: Theme 1 – Description: Documents (DOC001, DOC002)\n\n{joined}"
            themes = ask_gemini(theme_prompt)
        else:
            answer = ask_gemini(query)
            themes = "No theme found."

        st.markdown("### ✅ Answer")
        st.success(answer)

        if table_data:
            import pandas as pd
            st.markdown("### 📊 Matching Document Results")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        st.markdown("### 🧠 Theme Summary")
        st.info(themes)
