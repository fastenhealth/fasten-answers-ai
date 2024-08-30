import requests


class LlamaCppClient:
    def __init__(self, settings: dict):
        self.base_url = settings.get("host")
        self.model_prompt = settings.get("model_prompt")
        self.tokenizer = settings.get("tokenizer")
        self.system_prompt = settings.get("system_prompt")
        self.n_predict = settings.get("n_predict")
        self.temperature = settings.get("temperature")
        self.stop = settings.get("stop")
        self.stream = settings.get("stream")

    def chat(self, user_prompt: str, question: str):
        params = {
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "stop": self.stop,
            "stream": self.stream,
            "cache_prompt": False,
        }

        model_prompt = self.model_prompt.format(system_prompt=self.system_prompt, user_prompt=user_prompt, question=question)

        data = {"prompt": model_prompt, **params}

        response = requests.post(f"{self.base_url}/completion", json=data)
        return response
