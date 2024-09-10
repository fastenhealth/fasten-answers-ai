import requests

import tiktoken


class OpenAIHandler:
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        self.model = model

    def get_total_tokens_from_prompt(self, prompt: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        tokens = encoding.encode(prompt)
        return len(tokens)

    def get_chat_completion(
        self,
        openai_api_key,
        user_prompt: str,
        system_prompt: str,
        answer_json_schema: dict,
        max_tokens: int = 300,
        endpoint_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
        }
        if answer_json_schema:
            payload["response_format"] = answer_json_schema

        try:
            response = requests.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            print(f"An error occurred: {err}")
            return None

    def calculate_api_costs(
        self,
        total_input_tokens: int,
        total_openai_requests: int,
        cost_per_million_input_tokens: float,
        cost_per_million_output_tokens: float,
        tokens_per_response: int,
    ) -> dict:
        input_cost = round(total_input_tokens * (cost_per_million_input_tokens / 1_000_000), 3)
        total_output_tokens = total_openai_requests * tokens_per_response
        output_cost = round(total_output_tokens * (cost_per_million_output_tokens / 1_000_000), 3)
        total_cost = round(input_cost + output_cost, 3)

        return {
            "total_cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }
