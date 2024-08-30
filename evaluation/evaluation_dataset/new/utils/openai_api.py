import json
import numpy as np
from requests.exceptions import HTTPError
from typing import List, Dict

import openai
from create_chunks import get_total_tokens_from_string
from settings import QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL, OPENAI_MODEL, MAX_TOKENS, logger


class OpenAIBatchAPI:
    def __init__(
        self,
        system_prompt=QUESTION_GEN_SYS_TMPL,
        user_prompt=QUESTION_GEN_USER_TMPL,
        max_tokens=MAX_TOKENS,
        openai_model=OPENAI_MODEL,
        tokens_limit=10000,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_tokens = max_tokens
        self.openai_model = openai_model
        self.tokens_limit = tokens_limit
        self.results = []
        self.total_tokens = []

    def generate_qa_file_batch_api(
        self, resources: Dict[str, dict], num_questions_per_chunk: int = 1, output_file: str = "qa_file.jsonl"
    ) -> (int, int):
        """Generate questions and save to a .jsonl file."""
        method = "POST"
        url = "/v1/chat/completions"
        system_prompt_formatted = self.system_prompt.format(num_questions_per_chunk=num_questions_per_chunk)

        tokens_system_prompt = get_total_tokens_from_string(system_prompt_formatted)

        for key, value in resources.items():
            user_prompt_formatted = self.user_prompt.format(context_str=value.get("text_chunk"))
            tokens_user_prompt = get_total_tokens_from_string(user_prompt_formatted)

            if tokens_system_prompt + tokens_user_prompt < self.tokens_limit:
                input_object = {
                    "custom_id": key,
                    "method": method,
                    "url": url,
                    "body": {
                        "model": self.openai_model,
                        "messages": [
                            {"role": "system", "content": system_prompt_formatted},
                            {"role": "user", "content": user_prompt_formatted},
                        ],
                        "max_tokens": self.max_tokens,
                    },
                }
                self.results.append(input_object)
                self.total_tokens.append(tokens_user_prompt)

        total_system_tokens = tokens_system_prompt * len(self.results)
        self.total_tokens.append(total_system_tokens)

        # Save to .jsonl file
        self._save_to_jsonl(output_file)

        total_input_tokens = np.sum(self.total_tokens)
        total_openai_queries = len(self.results)

        return total_input_tokens, total_openai_queries

    def _save_to_jsonl(self, output_file: str) -> None:
        """Helper function to save results to a .jsonl file."""
        with open(output_file, "w") as f:
            for item in self.results:
                f.write(json.dumps(item) + "\n")

    def openai_create_completion(self, system_prompt, user_prompt) -> dict:
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=self.max_tokens,
            )
            return response
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")
        return None

    @staticmethod
    def get_aproximate_costs(
        total_tokens_input: int,
        total_openai_queries: int,
        cost_per_million_input: float = 0.075,
        cost_per_million_output: float = 0.3,
        tokens_generated: int = 300,
    ) -> (float, float, float):
        """Generate aprox costs of using Batch API"""
        input_costs = round(total_tokens_input * (cost_per_million_input / 1000000), 3)
        total_tokens_output = total_openai_queries * tokens_generated
        output_costs = round((total_tokens_output) * (cost_per_million_output / 1000000), 3)
        total_costs = round(input_costs + output_costs, 2)

        return total_costs, input_costs, output_costs
