import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from vector_store import VectorStoreConfig
import warnings

load_dotenv()

warnings.filterwarnings("ignore")


class RagBot:
    def __init__(self):
        csv_path = os.getenv("CSV_PATH")
        self.llm = ChatOllama(model="llama3")
        
        vector_store_config = VectorStoreConfig(csv_path=csv_path)
        self.vector_store = vector_store_config.load_or_create_vector_store()
        self.retriever = self.vector_store.as_retriever()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever = self.retriever,
            memory = self.memory
        )
        
    def ask(self, question: str) -> str:
        response = self.chain.invoke({"question": question})
        return response["answer"]