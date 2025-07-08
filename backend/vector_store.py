import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from load_data import load_data

load_dotenv()

class VectorStoreConfig:
    def __init__(self, vector_store_path: str = "faiss_medical_index", embedding_model: str = "mxbai-embed-large"):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        
    def vector_store_exists(self) -> bool:
        return os.path.exists(self.vector_store_path) and len(os.listdir(self.vector_store_path)) > 0

    def create_vector_store(self):
        docs = load_data()
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local(self.vector_store_path)

    def load_or_create_vector_store(self):
        if not self.vector_store_exists():
            print(f"Vector store not found at {self.vector_store_path}. Creating a new one...")
            self.create_vector_store()
            return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
