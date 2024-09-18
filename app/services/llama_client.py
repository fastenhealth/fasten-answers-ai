import requests
import json

import asyncio
import aiohttp
import traceback

from app.config.settings import settings, logger


class LlamaCppClient:
    DEFAULT_PARAMS = {
        "n_predict": 400,
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
        self.base_url = settings.model.llm_host

    async def _call_model_parallel(self, url, payloads) -> list[dict]:
        """This function calls the model at url in parallel with multiple payloads"""
        async with aiohttp.ClientSession() as session:

            async def make_request(url, data):
                async with session.post(url, json=data) as response:
                    return await response.json()

            return await asyncio.gather(*[make_request(url, data) for data in payloads])

    def _build_payload(self, model_prompt, query, params, context=None):
        if context:
            prompt = model_prompt.format(context=context, query=query)
        else:
            prompt = model_prompt.format(query=query)

        data = {"prompt": prompt, **params}

        return data

    def chat(self, query, stream, model_prompt: str, context=None, params=None) -> dict:
        params = params or self.DEFAULT_PARAMS.copy()
        params["stream"] = stream

        data = self._build_payload(model_prompt=model_prompt, query=query, params=params, context=context)

        try:
            if params["stream"]:
                return self._stream_response(data)
            else:
                return self._get_response(data)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    async def process_parallel(
        self, contexts=None, messages=None, resource_batch=None, model_prompt=None, params=None
    ) -> list[dict]:
        url = f"{self.base_url}/completion"
        params = params or self.DEFAULT_PARAMS.copy()

        payloads = []

        # Chat
        if contexts and messages:
            for context, message in zip(contexts, messages):
                payloads.append(self._build_payload(context, message, params))

        # Summarize
        elif resource_batch and model_prompt:
            for resource in resource_batch:
                payloads.append(self._build_payload(model_prompt=model_prompt, query=resource["resource"], params=params))

        logger.info(f"Sending parallel requests to llama.cpp server with {len(payloads)} payloads")

        try:
            return await self._call_model_parallel(url, payloads)
        except requests.RequestException as e:
            logger.error(f"Error processing batch: {str(e)}\n{traceback.format_exc()}")
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
