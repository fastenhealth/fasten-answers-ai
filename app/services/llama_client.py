import requests
import json

from ..config.settings import settings, logger


class LlamaCppClient:
    def __init__(self, base_url=settings.llama_host):
        self.base_url = base_url
        self.system_prompt = settings.llama_prompt
        self.conversation_history = []

    def chat(self, message, context, params=None):
        self.conversation_history.append(f"User: {message}")

        if params is None:
            params = {
                "n_predict": 400,
                "temperature": 0.7,
                "stop": ["</s>", "User:", "Assistant:"],
                "repeat_last_n": 256,
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

        prompt = f"{self.system_prompt}\n\n \
            Context: {context}\n\n" + "\n".join(self.conversation_history) + "\nAssistant:"

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

            # Update conversation_history with the full response
            self.conversation_history.append(f"Assistant: {full_content.strip()}")
        else:
            logger.error(f"Error: {response.status_code}, {response.text}")
            raise Exception(f"Error: {response.status_code}, {response.text}")


llm_client = LlamaCppClient()
