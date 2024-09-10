from elasticsearch import Elasticsearch

from app.config.settings import settings, logger


def get_es_client():
    return Elasticsearch(
        hosts=[settings.elasticsearch.host],
        basic_auth=(settings.elasticsearch.user, settings.elasticsearch.password),
        max_retries=10,
    )


def get_mapping():
    return {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 384},
                "metadata": {"type": "object"},
            }
        }
    }


def create_index_if_not_exists(index_name):
    es_client = get_es_client()
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=get_mapping())
        logger.info(f"Index '{index_name}' created.")
    return es_client
