import requests
from app.core.config import GROQ_API_KEY, GROQ_API_URL

def ask_groq(prompt, context_docs):
    context = "\n\n".join(context_docs)
    full_prompt = f"Using the context below, answer the question:\n\n{context}\n\nQ: {prompt}"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "Be concise and give citations."},
            {"role": "user", "content": full_prompt}
        ]
    }
    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=data)
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"API Error: {e}"
