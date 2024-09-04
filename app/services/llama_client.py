import asyncio
import aiohttp
import requests
import json
from typing import List

from ..config.settings import settings, logger


class LlamaCppClient:
    DEFAULT_PARAMS = {
        "n_predict": 200,
        "temperature": 0.0,
        "stop": ["<|end|>"],
        "repeat_last_n": 64,
        "repeat_penalty": 1.18,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "tfs_z": 1.0,
        "typical_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "stream": False,
    }

    def __init__(self):
        self.base_url = settings.host
        self.system_prompt = settings.system_prompt
        self.prompt = settings.model_prompt

    async def _call_model_parallel(self, url, payloads) -> List[dict]:
        """This function calls the model at url in parallel with multiple payloads"""
        async with aiohttp.ClientSession() as session:

            async def fetch(url, data):
                async with session.post(url, json=data) as response:
                    return await response.json()

            return await asyncio.gather(*[fetch(url, data) for data in payloads])

    def _build_payload(self, context, message, params):
        prompt = self.prompt.format(system_prompt=self.system_prompt, context=context, message=message)

        data = {"prompt": prompt, **params}

        return data

    def chat(self, context, message, stream, params=None) -> dict:
        params = params or self.DEFAULT_PARAMS.copy()
        params["stream"] = stream

        data = self._build_payload(context, message, params)
        # logger.info(f"Sending request to llama.cpp server with prompt: {prompt}")

        try:
            if params["stream"]:
                return self._stream_response(data)
            else:
                return self._get_response(data)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def chat_parallel(self, contexts, messages, params=None) -> List[dict]:
        params = params or self.DEFAULT_PARAMS.copy()

        payloads = []
        for context, message in zip(contexts, messages):
            payloads.append(self._build_payload(context, message, params))

        logger.info(f"Sending parallel requests to llama.cpp server with {len(payloads)} payloads")
        url = f"{self.base_url}/completion"
        try:
            return asyncio.run(self._call_model_parallel(url, payloads))
        except requests.RequestException as e:
            logger.error(f"Parallel requests failed: {e}")
            raise

    def _stream_response(self, data):
        response = requests.post(f"{self.base_url}/completion", json=data, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    content = json.loads(decoded_line[6:])
                    if content.get("stop"):
                        break
                    chunk = content.get("content", "")
                    yield chunk

    def _get_response(self, data):
        response = requests.post(f"{self.base_url}/completion", json=data)
        response.raise_for_status()
        return response.json()


llm_client = LlamaCppClient()
