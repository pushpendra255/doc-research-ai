# ✅ EduMentor: Streamlit-only PDF Chatbot with Gemini API (Free Version Fixed)
# Final version for Wasserstoff Gen-AI Internship Task

import streamlit as st
from PyPDF2 import PdfReader
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pandas as pd

# 🔐 Gemini API Key Configuration (Free API Key)
genai.configure(api_key="AIzaSyBeoYwJuJSaOGyWbNwzgoGl8rb2OtctSN8")

# 🔎 Load Sentence Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Let it auto-select CPU

# 📄 Text Extraction from PDF
def extract_text(file):
    try:
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

# 🧠 Semantic Search
def get_most_similar_docs(query, texts, top_k=3):
    query_vec = model.encode([query])
    text_vecs = model.encode(texts)
    sims = cosine_similarity(query_vec, text_vecs)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [texts[i] for i in top_indices]

# 🤖 Ask Gemini (Free API Compatible)
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-pro")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 512
            }
        )
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

# 📍 Get Citation
def get_citation(text, query):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if query.lower() in line.lower():
            return f"Page {i//25 + 1}, Line {i%25 + 1}"
    return "Not Found"

# 🧠 Streamlit App UI
st.set_page_config(page_title="EduMentor – Gemini Chatbot", layout="wide")
st.title("📘 EduMentor – Theme-Based PDF Chatbot (Gemini AI)")

uploaded_files = st.file_uploader("📄 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if any([f is not None and hasattr(f, 'read') for f in uploaded_files]):
        st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully.")
    else:
        st.error("❌ File upload failed. Try another browser or file.")
        st.stop()
else:
    st.info("📂 Please upload at least one PDF to continue.")

query = st.text_input("🔍 Ask a question:", placeholder="Example: What are the key goals of the NEP?")
submit = st.button("✍️ Get Answer")

if submit and query:
    if not uploaded_files:
        st.warning("⚠️ Upload at least one PDF first.")
        st.stop()

    with st.spinner("🔎 Processing documents..."):
        doc_texts, doc_info = [], []

        for i, file in enumerate(uploaded_files):
            doc_id = f"DOC{i+1:03d}"
            text = extract_text(file)
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

            final_answer = ask_gemini(
                f"Answer the following question from the document snippets:\n\n{joined}\n\nQ: {query}"
            )

            theme_summary = ask_gemini(
                f"Identify themes from these document snippets. Format:\nTheme 1 – Description: Documents (DOC001, DOC002)\n\n{joined}"
            )
        else:
            final_answer = ask_gemini(query)
            theme_summary = "No theme found."

        st.markdown("### ✅ Answer")
        st.success(final_answer)

        if table_data:
            st.markdown("### 📊 Matching Document Results")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        st.markdown("### 🧠 Theme Summary")
        st.info(theme_summary)
