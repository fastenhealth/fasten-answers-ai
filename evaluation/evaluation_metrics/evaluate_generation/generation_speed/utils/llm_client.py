from .settings import MODEL_SETTINGS, SYSTEM_PROMPT

from .llm import LlamaCppClient


def configure_llm_client(params: dict) -> LlamaCppClient:
    model_name = params.get("model")
    settings = {
        "host": params.get("host", "http://localhost:8090"),
        "model_prompt": MODEL_SETTINGS[model_name]["model_prompt"],
        "tokenizer": MODEL_SETTINGS[model_name]["tokenizer"],
        "system_prompt": SYSTEM_PROMPT,
        "total_cores": params.get("total_cores", 10),
        "n_predict": params.get("tokens_to_predict", 400),
        "temperature": params.get("temperature", 0.8),
        "stop": ["<|eot_id|>"],
        "stream": False,
    }

    return LlamaCppClient(settings)
