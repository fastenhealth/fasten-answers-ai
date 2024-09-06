from sentence_transformers import SentenceTransformer
from app.config.settings import settings


def get_sentence_transformer():
    return SentenceTransformer(settings.model.embedding_model_name)
