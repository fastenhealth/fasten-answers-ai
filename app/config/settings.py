import logging
import os


class ElasticsearchSettings:
    def __init__(self):
        self.host = os.getenv("ES_HOST", "http://localhost:9200")
        self.user = os.getenv("ES_USER", "elastic")
        self.password = os.getenv("ES_PASSWORD", "changeme")
        self.index_name = os.getenv("INDEX_NAME", "fasten-index")


class ModelsSettings:
    def __init__(self):
        # Base dir
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Embedding model
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        # LLM host
        self.llm_host = os.getenv("LLAMA_HOST", "http://localhost:9090")
        # Conversation prompts
        self.conversation_model_prompt = {
            "llama3.1": self.load_prompt(os.path.join(
                base_dir, "prompts/conversation_model_prompt_llama3.1-instruct.txt")),
            "Phi3.5-mini": self.load_prompt(os.path.join(
                base_dir, "prompts/conversation_model_prompt_Phi-3.5-instruct.txt"))
        }
        # Summaries prompts
        self.summaries_model_prompt = self.load_prompt(
            os.path.join(
                base_dir, "prompts/summaries_model_prompt_llama3.1-instruct.txt")
        )
        # Summaries openai system prompt
        self.summaries_openai_system_prompt = self.load_prompt(
            os.path.join(
                base_dir, "prompts/summaries_openai_system_prompt.txt")
        )

    def load_prompt(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            return file.read().strip().replace("\n", " ")


class Settings:
    def __init__(self):
        self.elasticsearch = ElasticsearchSettings()
        self.model = ModelsSettings()


settings = Settings()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
