import json

from ..config.settings import settings, logger
from ..config.profiling import profile


@profile
def search_query(query_text, embedding_model,
                 es_client, index_name=settings.index_name,
                 k=5, threshold=0.2,
                 text_boost=1.0, embedding_boost=1.0):
    logger.info(f"Searching for query: {query_text}")
    query_embedding = embedding_model.encode(query_text).tolist()
    query_body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                                "boost": text_boost
                            }
                        }
                    },
                    {
                        "script_score": {
                            "query": {
                                "match_all": {}
                            },
                            "script": {
                                "source": """
                                double score = cosineSimilarity(params.query_vector, 'embedding');
                                if (score < 0) {
                                    return 0;
                                }
                                return score;
                                """,
                                "params": {
                                    "query_vector": query_embedding
                                }
                            },
                            "boost": embedding_boost
                        }
                    }
                ]
            }
        },
        "_source": ["content", "metadata"]
    }
    response = es_client.search(index=index_name, body=query_body)
    results = response['hits']['hits']
    filtered_results = [result for result in results
                        if result['_score'] >= threshold]
    logger.info(f"Found {len(filtered_results)} \
        results for query: {query_text}")
    return [{"score": result['_score'],
             "content": str(result['_source']['content']),
             "metadata": result['_source'].get('metadata', {})}
            for result in filtered_results]
