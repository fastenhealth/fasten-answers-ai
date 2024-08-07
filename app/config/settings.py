import logging
import os


class Settings:
    es_host: str = os.getenv("ES_HOST", "http://localhost:9200")
    es_user: str = os.getenv("ES_USER", "elastic")
    es_password: str = os.getenv("ES_PASSWORD", "changeme")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME",
                                          "all-MiniLM-L6-v2")
    llama_host: str = os.getenv("LLAMA_HOST", "http://localhost:8080")
    llama_prompt: str = os.getenv("LLAMA_PROMPT",
                                 ( "A chat between a curious user and an intelligent, "
                                   "polite medical assistant. The assistant provides detailed, "
                                   "helpful answers to the user's medical questions, "
                                   "including accurate references where applicable."))
    index_name: str = os.getenv("INDEX_NAME", "fasten-index")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/")


settings = Settings()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
