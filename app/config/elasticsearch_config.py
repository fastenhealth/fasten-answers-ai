from elasticsearch import Elasticsearch

from .settings import settings


def get_es_client():
    return Elasticsearch(
        hosts=[settings.es_host],
        basic_auth=(settings.es_user, settings.es_password),
        max_retries=10,
    )
