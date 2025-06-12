import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
doc_texts = []
doc_ids = []

def store_documents(docs):
    for doc_id, text in docs:
        doc_ids.append(doc_id)
        doc_texts.append(text)
        vec = model.encode([text])
        index.add(vec)

def search_documents(query):
    q_vec = model.encode([query])
    D, I = index.search(q_vec, k=5)
    return [doc_texts[i] for i in I[0] if i < len(doc_texts)]
