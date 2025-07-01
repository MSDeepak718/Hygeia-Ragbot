from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import warnings

warnings.filterwarnings("ignore")

def load_rag_chain():
    llm = ChatOllama(model='llama3')
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    vector_store = FAISS.load_local('faiss_medical_index', embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain

if __name__ == "__main__":
    chain = load_rag_chain()
    print('\n')
    print("Welcome to the medical assistant!")
    while True:
        print("---------------------------------------------------------------------------------------------------------------------------------------\n")
        question = input('Ask your medical question ( or \'q\' to exit ): ')
        if question.strip().lower() == 'q':
            print('Thank you, Feel free to contact me any time!')
            break
        answer = chain.invoke({"query": question})
        print(f"\nResponse: {answer['result']}\n")
