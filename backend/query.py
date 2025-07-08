from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from vector_store import VectorStoreConfig
import warnings

warnings.filterwarnings("ignore")


class RagBot:
    def __init__(self):
        self.llm = ChatOllama(model="llama3")
        vector_store_config = VectorStoreConfig()
        self.vector_store = vector_store_config.load_or_create_vector_store()
        self.retriever = self.vector_store.as_retriever()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever = self.retriever,
            memory = self.memory
        )
        
    def ask(self, question: str) -> str:
        context = """
        You are a medical expert'
        Greet the user politely and ask them to provide their medical question if they used any greeting words.
        Don't mention the word context or reference the data source in your answer.
        Use the information from the context to provide a concise and accurate answer.
        If the question is not related to the context, say 'I don't know and suggest them to consult their doubt with a trained medical professional'.
        Ask a series of clarifying questions to narrow down the patient's condition if the question is vague or too broad.
        If the question is about a specific medical condition, provide a detailed answer based on the context.
        """
        question = f"{context} {question}"
        response = self.chain.invoke({"question": question})
        return response["answer"]