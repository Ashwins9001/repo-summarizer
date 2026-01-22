from langchain_community.vectorstores import FAISS
from embeddings import get_embeddings
import os

PATH = "data/faiss"

def load_store():
    if os.path.exists(PATH):
        return FAISS.load_local(
            PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None

def save_store(docs):
    store = FAISS.from_documents(docs, get_embeddings())
    store.save_local(PATH)
    return store
