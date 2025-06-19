# EduMentor v2: Per-Document QA + Clustering-Based Theme Detection using Groq API

import streamlit as st
from PyPDF2 import PdfReader
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import requests
import json

# Groq API Configuration
groq_api_key = "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"
groq_url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {groq_api_key}",
    "Content-Type": "application/json"
}

# Load Sentence Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Text Extraction from PDF
def extract_text(file):
    try:
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

# Ask Groq (LLaMA 3)
def ask_groq(prompt):
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    try:
        res = requests.post(groq_url, headers=headers, data=json.dumps(data))
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Groq API Error: {e}"

# Get Citation
def get_citation(text, query):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if query.lower() in line.lower():
            return f"Page {i//25 + 1}, Line {i%25 + 1}"
    return "Not Found"

# Cluster documents by themes
def cluster_documents(texts, n_clusters=3):
    embeddings = model.encode(texts)
    kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append((f"DOC{i+1:03d}", texts[i]))
    return clusters

# Streamlit App UI
st.set_page_config(page_title="EduMentor – Groq Chatbot", layout="wide")
st.title("EduMentor – Theme-Based PDF Chatbot (Groq LLaMA-3)")

uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if any([f is not None and hasattr(f, 'read') for f in uploaded_files]):
        st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully.")
    else:
        st.error("File upload failed. Try another browser or file.")
        st.stop()
else:
    st.info("Please upload at least one PDF to continue.")

query = st.text_input("Ask a question:", placeholder="Example: What are the key goals of the NEP?")
submit = st.button("Get Answer")

if submit and query:
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
        st.stop()

    with st.spinner("Processing documents..."):
        doc_texts, doc_info = [], []

        for i, file in enumerate(uploaded_files):
            doc_id = f"DOC{i+1:03d}"
            text = extract_text(file)
            doc_texts.append(text)
            doc_info.append((doc_id, text, file.name))

        matched_docs_display = []
        table_data = []

        for doc_id, text, name in doc_info:
            prompt = f"Answer this question based only on the document below:\n\n{text[:3000]}\n\nQ: {query}"
            answer_text = ask_groq(prompt)
            citation = get_citation(text, query)
            matched_docs_display.append(f"{doc_id}: {answer_text}")
            table_data.append({
                "Document ID": doc_id,
                "Extracted Answer": answer_text,
                "Citation": citation,
                "Source File": name
            })

        if matched_docs_display:
            joined = "\n".join(matched_docs_display)

            final_answer = ask_groq(
                f"Answer this question using the following document-based answers. Give a short, complete summary.\n\n{joined}\n\nQ: {query}"
            )

            clusters = cluster_documents(doc_texts, n_clusters=3)
            theme_summary = ""
            for cluster_id, docs in clusters.items():
                texts = "\n\n".join([doc[1][:1000] for doc in docs])
                theme = ask_groq(
                    f"Identify a common theme for these documents:\n{texts}\n\nFormat: Theme {cluster_id+1} – Description."
                )
                ids = ", ".join([doc[0] for doc in docs])
                theme_summary += f"\n{theme}\nDocuments: {ids}\n"
        else:
            final_answer = ask_groq(query)
            theme_summary = "No theme found."

        st.markdown("### ✅ Answer")
        st.success(final_answer)

        if table_data:
            st.markdown("### 📊 Matching Document Results")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        st.markdown("### 🧠 Theme Summary")
        st.info(theme_summary)