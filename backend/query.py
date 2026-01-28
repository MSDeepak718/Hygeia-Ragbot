from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_classic.memory import ConversationBufferMemory
from vector_store import VectorStoreConfig
from hybrid_retriever import HybridRRFRetriever
import warnings

warnings.filterwarnings("ignore")


class RagBot:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOllama(model="llama3.1:8b")

        # Load vector store
        vector_store_config = VectorStoreConfig()
        self.vector_store = vector_store_config.load_or_create_vector_store()
        docs = vector_store_config.get_documents()
        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        hybrid_retriever = HybridRRFRetriever(
            dense_retriever=dense_retriever,
            documents=docs,
            k_dense=5,
            k_sparse=5,
        )
        self.retriever = RunnableLambda(lambda q: hybrid_retriever.invoke(q))

        # Memory for chat history
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Prompt Template
        self.prompt = ChatPromptTemplate.from_template("""
        You are a medical expert.
        Greet the user politely and ask them to provide their medical question if they used any greeting words.
        Don't mention the word 'context' or reference the data source in your answer.
        Use the information from the context to provide a concise and accurate answer.
        If the question is not related to the context, say "I don't know" and suggest they consult a trained medical professional.
        Ask clarifying questions if the query is vague or too broad.

        Context:
        {context}

        Question:
        {question}
        """)

        # Build modern RAG pipeline
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

    def ask(self, question: str) -> str:
        response = self.chain.invoke(question)
        return response.content if hasattr(response, "content") else str(response)
