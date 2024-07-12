from sentence_transformers import SentenceTransformer
from ..config.settings import settings


def get_sentence_transformer():
    return SentenceTransformer(settings.embedding_model_name)
