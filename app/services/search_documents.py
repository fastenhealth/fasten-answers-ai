from app.config.settings import settings, logger


def search_query(query_text,
                 embedding_model,
                 es_client,
                 index_name=settings.elasticsearch.index_name,
                 k=5,
                 text_boost=0.25,
                 embedding_boost=4.0):
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
    logger.info(f"Found {len(results)} results for the query: {query_text}")
    return [{"score": result['_score'],
             "content": str(result['_source']['content']),
             "metadata": result['_source'].get('metadata', {})}
            for result in results]


def fetch_all_documents(es_client,
                      index_name=settings.elasticsearch.index_name):
    query_body = {
        "query": {
            "match_all": {}
        },
        "_source": ["content", "metadata"]
    }

    response = es_client.search(index=index_name, body=query_body)
    results = response['hits']['hits']

    return [{"id": result['_id'],
             "content": result['_source'].get('content', ''),
             "metadata": result['_source'].get('metadata', {})}
            for result in results]
