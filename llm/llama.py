import requests
import json

class Settings:
    llama_host = "http://localhost:8080"
    llama_prompt = ("You are an intelligent and polite medical assistant "
                    "who provides detailed and "
                    "helpful answers to user's medical questions, "
                    "including accurate references where applicable.")

settings = Settings()

class LlamaCppClient:
    def __init__(self, base_url=settings.llama_host):
        self.base_url = base_url
        self.llama_prompt = settings.llama_prompt
        self.conversation_history = []

    def chat(self, message, params=None):
        if params is None:
            params = {
                "n_predict": 400,
                "temperature": 0.7,
                "stop": ["<|eot_id|>"],
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
                "n_keep": 33,
                "stream": True
            }

        

        # Construir el prompt con la historia de la conversación
        prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                  f"{self.llama_prompt}<|eot_id|>\n"
                  f"<|start_header_id|>user<|end_header_id|>\n\n"
                  f"{message}<|eot_id|>\n"
                  "<|start_header_id|>assistant<|end_header_id|>"
                  )


        data = {
            "prompt": prompt,
            **params
        }

        response = requests.post(f"{self.base_url}/completion",
                                 json=data, stream=True)

        if response.status_code == 200:
            full_content = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        content = json.loads(decoded_line[6:])
                        chunk = content['content']
                        full_content += chunk
                        yield chunk
                        if content['stop']:
                            break
        else:
            print(f"Error: {response.status_code}, {response.text}")
            raise Exception(f"Error: {response.status_code}, {response.text}")

llm_client = LlamaCppClient()