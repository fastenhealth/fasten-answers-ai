from typing import List
import aiohttp
import asyncio


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

    async def _call_model(self, url, pl):
        async with aiohttp.ClientSession() as session:

            async def fetch(url, data, i):
                async with session.post(url, json=data) as response:
                    j = await response.json()
                    return j

            return await asyncio.gather(*[fetch(url, data, i) for i, data in enumerate(pl)])

    def chat(self, user_prompts: List[str], questions: List[str]):
        params = {
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "stop": self.stop,
            "stream": self.stream,
            "cache_prompt": False,
        }
        pl = []
        for user_prompt, question in zip(user_prompts, questions):
            model_prompt = self.model_prompt.format(system_prompt=self.system_prompt, user_prompt=user_prompt, question=question)

            data = {"prompt": model_prompt, **params}
            pl.append(data)

        url = f"{self.base_url}/completion"
        responses = asyncio.run(self._call_model(url, pl))

        return responses
