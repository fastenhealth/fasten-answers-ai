from typing import List

from app import reranker
from app.config.settings import settings
from app.data_models.search_result import SearchResult
from app.services.reranking import RerankingService

def search_query(
    query_text,
    embedding_model,
    es_client,
    index_name=settings.elasticsearch.index_name,
    k=5,
    text_boost=0.25,
    embedding_boost=4.0,
    rerank_top_k=0
) -> List[SearchResult]:
    query_embedding = embedding_model.encode(query_text,
                                             show_progress_bar=False).tolist()
    query_body = {
        "size": max(k, rerank_top_k),
        "query": {
            "bool": {
                "should": [
                    {"match": {"content": {"query": query_text, "boost": text_boost}}},
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": """
                                double score = cosineSimilarity(params.query_vector, 'embedding');
                                if (score < 0) {
                                    return 0;
                                }
                                return score;
                                """,
                                "params": {"query_vector": query_embedding},
                            },
                            "boost": embedding_boost,
                        }
                    },
                ]
            }
        },
        "_source": ["content", "metadata"],
    }
    response = es_client.search(index=index_name, body=query_body)
    results = response["hits"]["hits"]
    search_results = [
        SearchResult(score=result["_score"],
                     content=str(result["_source"]["content"]),
                     metadata=result["_source"].get("metadata", {}),
        )
        for result in results
    ]
    if rerank_top_k > 0:
        reranker: RerankingService = reranker
        search_results = [result for result, score in reranker.rerank(query_text, search_results)[:k]]
    return search_results


def fetch_all_documents(es_client,
                        index_name=settings.elasticsearch.index_name,
                        size: int = 2000):
    query_body = {"query":
                  {
                      "match_all": {}
                  },
                  "_source": ["content", "metadata"],
                  "size": size
                  }

    response = es_client.search(index=index_name, body=query_body)
    results = response["hits"]["hits"]

    return [
        {"id": result["_id"], "content": result["_source"].get(
            "content", ""), "metadata": result["_source"].get("metadata", {})}
        for result in results
    ]
