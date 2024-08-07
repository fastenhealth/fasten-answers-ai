import requests
import json

from ..config.settings import settings, logger
from ..config.profiling import profile


class LlamaCppClient:
    def __init__(self, base_url=settings.llama_host):
        self.base_url = base_url
        self.system_prompt = settings.llama_prompt

    @profile
    def chat(self, message, context, params=None):

        if params is None:
            params = {
                "n_predict": 400,
                "temperature": 0.8,
                "stop": ["<|eot_id|>"],
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
                "stream": True
            }
        
        prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                  f"{self.system_prompt}<|eot_id|>\n"
                  f"<|start_header_id|>user<|end_header_id|>\n\n"
                  "Context information is below.\n"
                  "---------------------\n"
                  f"{context}\n"
                  "---------------------\n"
                  "Given the context information (if there is any), "
                  "this is my message: "
                  f"{message}<|eot_id|>\n"
                  "<|start_header_id|>assistant<|end_header_id|>"
                  )

        data = {
            "prompt": prompt,
            **params
        }

        logger.info(f"Sending request to Llama server with prompt: {prompt}")

        response = requests.post(f"{self.base_url}/completion",
                                 json=data, stream=True)


        if response.status_code == 200:
            full_content = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        content = json.loads(decoded_line[6:])
                        if content['stop']:
                            break
                        chunk = content['content']
                        full_content += chunk
                        yield chunk

        else:
            logger.error(f"Error: {response.status_code}, {response.text}")
            raise Exception(f"Error: {response.status_code}, {response.text}")


llm_client = LlamaCppClient()
