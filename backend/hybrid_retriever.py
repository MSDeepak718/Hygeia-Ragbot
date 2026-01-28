from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
from collections import defaultdict

class HybridRRFRetriever:
    def __init__(
        self,
        dense_retriever,
        documents: List[Document],
        k_dense: int = 5,
        k_sparse: int = 5,
        rrf_k: int = 60,
    ):
        """
        Docstring for __init__
        :param self: Description
        :param dense_retriever: Description
        :param documents: Description
        :type documents: List[Document]
        :param k_dense: Description
        :type k_dense: int
        :param k_sparse: Description
        :type k_sparse: int
        :param rrf_k: Description
        :type rrf_k: int
        """

        self.dense_retriever = dense_retriever
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = k_sparse

    def __rrf_fusion(self, ranked_lists: List[List[Document]]) -> [Document]:
        scores = defaultdict(float)
        for docs in ranked_lists:
            for rank, doc in enumerate(docs):
                doc_id = doc.page_content
                scores[doc_id] += 1/(self.rrf_k + rank +1)
        
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        content_to_doc = {}
        for docs in ranked_lists:
            for doc in docs:
                content_to_doc[doc.page_content] = doc

        return [content_to_doc[doc_id] for doc_id, _ in sorted_docs]
    
    def invoke(self, query: str) -> List[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25.invoke(query)

        return self.__rrf_fusion([dense_docs, sparse_docs])

