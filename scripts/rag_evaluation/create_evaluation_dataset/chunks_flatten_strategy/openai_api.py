import json
import numpy as np
from requests.exceptions import HTTPError
from typing import List

import openai

from create_chunks import get_total_tokens_from_string
from settings import QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL, \
    OPENAI_MODEL, MAX_TOKENS, logger


def generate_qa_file_batch_api(
    resources: dict,
    system_prompt=QUESTION_GEN_SYS_TMPL,
    user_prompt=QUESTION_GEN_USER_TMPL,
    max_tokens=MAX_TOKENS,
    num_questions_per_chunk: int = 1,
    output_file: str = "qa_file.jsonl",
    tokens_limit: int = 10000
) -> None:
    """Generate questions and save to a .jsonl file."""
    results = []
    total_tokens = []

    method = "POST"
    url = "/v1/chat/completions"
    system_prompt = system_prompt.format(
        num_questions_per_chunk=num_questions_per_chunk)

    tokens_system_prompt = get_total_tokens_from_string(system_prompt)

    for key, value in resources.items():
        user_prompt_formatted = user_prompt.format(
            context_str=value.get("text_chunk")
        )
        tokens_user_prompt = get_total_tokens_from_string(user_prompt_formatted)

        if tokens_system_prompt + tokens_user_prompt < tokens_limit:
            input_object = {
                "custom_id": key,
                "method": method,
                "url": url,
                "body": {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_formatted}
                    ],
                    "max_tokens": max_tokens
                }
            }
            results.append(input_object)
            total_tokens.append(tokens_user_prompt)

    total_system_tokens = tokens_system_prompt * len(results)
    total_tokens.append(total_system_tokens)

    # Save to .jsonl file
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    total_input_tokens = np.sum(total_tokens)
    total_openai_queries = len(results)

    return total_input_tokens, total_openai_queries


def openai_create_completion(system_prompt,
                             user_prompt,
                             max_tokens=MAX_TOKENS,
                             OpenAI_model=OPENAI_MODEL):
    try:
        response = openai.chat.completions.create(
            model=OpenAI_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens
        )
        return response
    except HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred: {err}")
    return None


def aprox_costs(
    total_tokens_input: int,
    total_openai_queries: int,
    cost_per_million_input: float = 0.075,
    cost_per_million_output: float = 0.3,
    tokens_generated: int = 300
) -> None:
    """Generate aprox costs of using Batch API"""

    input_costs = round(total_tokens_input * (cost_per_million_input / 1000000), 3)

    total_tokens_output = total_openai_queries * tokens_generated
    output_costs = round((total_tokens_output) * (cost_per_million_output / 1000000), 3)

    total_costs = round(input_costs + output_costs, 2)

    return total_costs, input_costs, output_costs
