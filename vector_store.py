from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from load_data import load_data
import os
from dotenv import load_dotenv

load_dotenv()

csv_path = os.getenv('csv_path')

def create_vector_store():
    docs = load_data(csv_path)
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local('faiss_medical_index')

if __name__ == "__main__":
    create_vector_store()
