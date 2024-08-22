import requests
import json


class LlamaCppClient:
    def __init__(self, settings: dict):
        self.base_url = settings.get("host")
        self.model_prompt = settings.get("model_prompt")
        self.system_prompt = settings.get("system_prompt")
        self.n_predict = settings.get("n_predict")
        self.temperature = settings.get("temperature")
        self.stop = settings.get("stop")
        self.stream = settings.get("stream")

    def chat(self, user_prompt):
        params = {
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "stop": self.stop,               
            "stream": self.stream,
            "cache_prompt": False
        }

        model_prompt = self.model_prompt.format(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt
        )

        data = {
            "prompt": model_prompt,
            **params
        }

        response = requests.post(f"{self.base_url}/completion",
                                 json=data)
        return response
        #), stream=True)


        # if response.status_code == 200:
        #     full_content = ""
        #     for line in response.iter_lines():
        #         if line:
        #             decoded_line = line.decode('utf-8')
        #             if decoded_line.startswith('data: '):
        #                 content = json.loads(decoded_line[6:])
        #                 chunk = content['content']
        #                 full_content += chunk
        #                 yield chunk
        #                 if content['stop']:
        #                     break
        # else:
        #     print(f"Error: {response.status_code}, {response.text}")
        #     raise Exception(f"Error: {response.status_code}, {response.text}")

# llm_client = LlamaCppClient()