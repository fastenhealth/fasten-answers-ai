from FlagEmbedding import FlagReranker
from typing import List, Tuple

from app.data_models.search_result import SearchResult


class RerankingService:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.reranker = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, documents: List[SearchResult]) -> List[Tuple[SearchResult, float]]:
        """Computes a score for each document in the list of documents and returns a ranked list of documents.

        :param str query: user query used for ranking
        :param List[str] documents: Documents to be ranked
        :return tuple(str, float): list of tuples containing the document and its score
        """
        scores = self.reranker.compute_score([[query, doc.content] for doc in documents], normalize=True)

        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked_docs
