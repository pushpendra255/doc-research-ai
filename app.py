import streamlit as st
import requests

st.title("EduMentor – Chatbot")
uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
question = st.text_input("Ask your question")
if st.button("Submit"):
    if uploaded:
        res = requests.post("http://localhost:8000/upload", files=[("files", (f.name, f, "application/pdf")) for f in uploaded])
    ans = requests.get("http://localhost:8000/query", params={"q": question})
    st.success(ans.json().get("answer"))
