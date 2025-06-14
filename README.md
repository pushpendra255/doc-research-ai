# 📘 EduMentor – AI-Powered Policy Chatbot (Streamlit + Groq LLaMA 3)

An intelligent chatbot that allows users to upload 75+ government policy PDFs and ask questions based on them. The bot finds the most relevant document answers, gives proper citations, and synthesizes policy themes – all powered by LLaMA 3 via Groq API.

## 🚀 Features

- ✅ Upload and analyze 75+ policy PDFs
- ✅ Extract relevant document answers using semantic search
- ✅ Accurate citation (page + line)
- ✅ Synthesized theme identification
- ✅ Uses Groq LLaMA-3 for answering + summarizing
- ✅ Fully built using Streamlit (No backend required)
- ✅ Supports scanned & text-based PDFs

## 🧠 How It Works

1. Upload one or more PDFs (75+ supported)
2. Ask any question in plain English (e.g. What are the goals of NEP?)
3. The app:
   - Extracts answers from matching PDFs
   - Shows exact citation and file name
   - Uses Groq (LLaMA 3) to generate short answer + themes
4. Displays all results + themes clearly

## 🛠️ Tech Stack

- Python
- Streamlit
- sentence-transformers (for semantic similarity)
- PyPDF2 (for PDF parsing)
- Groq API (LLaMA-3 for final answers)

## 📂 Folder Structure

.
├── app.py                 # Streamlit frontend app  
├── requirements.txt       # Python dependencies  
└── packages.txt (optional)# For forcing Python version on Streamlit  

## 🔐 API Key Setup

Set your Groq API Key inside app.py:
groq_api_key = "YOUR_GROQ_API_KEY"

You can get one from: https://console.groq.com

## 🧪 Example Questions to Ask

- What are the highlights of the National Education Policy?
- What schemes are mentioned for rural development?
- List themes of digital governance from these documents.

## ✅ Live Demo

👉 [Click to View Live App]([https://your-username-doc-research-theme-idtif-chatbot.streamlit.app](https://doc-research-theme-idtif-chatbot.streamlit.app/])

## 📌 Assignment Requirement Mapping

| Requirement                             | Status |
|----------------------------------------|--------|
| Upload & analyze 75+ PDFs              | ✅ Done via Streamlit uploader |
| Document-based answer extraction       | ✅ Semantic Search + Citation |
| Theme identification                   | ✅ LLaMA 3 (Groq) used |
| Short, clean answers with source       | ✅ Displayed in table |
| Fully working, modern Streamlit app    | ✅ All included |

## 🙋‍♂️ Made by

Pushpendra Bhadauriya  
For Wasserstoff Gen-AI Internship Assignment  
