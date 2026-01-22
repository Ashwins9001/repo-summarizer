# embeddings_faiss.py
import faiss
import numpy as np
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class VectorStore:
    def __init__(self, persist_dir="data/faiss_index"):
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.docs_path = os.path.join(persist_dir, "docs.pkl")
        os.makedirs(persist_dir, exist_ok=True)

        self.embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.docs = []

    def build_from_docs(self, docs: list[Document]):
        # Split docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = []
        vectors = []

        for doc in docs:
            for chunk in splitter.split_text(doc.page_content):
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
                vec = np.array(self.embeddings_model.embed_text(chunk), dtype='float32')
                vectors.append(vec)

        self.docs = split_docs
        self.index = faiss.IndexFlatL2(vectors[0].shape[0])
        self.index.add(np.stack(vectors))

        # Save
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "wb") as f:
            pickle.dump(split_docs, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.docs_path):
            with open(self.docs_path, "rb") as f:
                self.docs = pickle.load(f)

    def search(self, query: str, k=5):
        vec = np.array(self.embeddings_model.embed_text(query), dtype='float32').reshape(1, -1)
        D, I = self.index.search(vec, k)
        return [self.docs[i] for i in I[0]]
