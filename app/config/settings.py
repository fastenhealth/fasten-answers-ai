import logging
import os


class Settings:
    es_host: str = os.getenv("ES_HOST", "http://localhost:9200")
    es_user: str = os.getenv("ES_USER", "elastic")
    es_password: str = os.getenv("ES_PASSWORD", "changeme")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME",
                                          "all-MiniLM-L6-v2")
    host: str = os.getenv("LLAMA_HOST", "http://localhost:8090")
    system_prompt: str = os.getenv("LLAMA_PROMPT",
                                   ("You are an intelligent, polite medical assistant embedded "
                                    "within a Retrieval-Augmented Generation (RAG) system. "
                                    "Your responses are based strictly on information retrieved "
                                    "from a database, specifically FHIR data chunks. These chunks may not always be clear. "
                                    "If you do not find relevant information, acknowledge it and do not attempt to fabricate answers. "
                                    "Provide detailed, helpful, and accurate responses, and include references where applicable. "
                                    "If information is not available, politely inform the user that you cannot provide an answer."))
    index_name: str = os.getenv("INDEX_NAME", "fasten-index")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/")
    model_prompt: str = ("<|system|>"
                         "{system_prompt}<|end|>"
                         "<|user|>"
                         "Context information is below.\n "
                         "---------------------\n "
                         "{context}\n"
                         "---------------------\n "
                         "Given the context information (if there is any), "
                         "this is my message: "
                         "{message}"
                         "<|end|>"
                         "<|assistant|>"
                         )


settings = Settings()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
