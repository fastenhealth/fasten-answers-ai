import json
import matplotlib.pyplot as plt
import numpy as np
from requests.exceptions import HTTPError
import secrets

import openai

from create_chunks import get_total_tokens_from_string
from settings import QUESTION_GEN_SYS_TMPL_1, QUESTION_GEN_USER_TMPL, \
    OPENAI_MODEL, MAX_TOKENS, logger


def generate_unique_id(length=4):
    """Generate a unique ID of the specified length."""
    return secrets.token_hex(length // 2)  # 4 characters


def generate_qa_file_batch_api(
    resources: list,
    system_prompt=QUESTION_GEN_SYS_TMPL_1,
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

    tokens_system_prompt = get_total_tokens_from_string(system_prompt)

    for resource in resources:
        user_prompt_formatted = user_prompt.format(
            context_str=resource.text
        )
        tokens_user_prompt = get_total_tokens_from_string(user_prompt_formatted)

        if tokens_system_prompt + tokens_user_prompt < tokens_limit:
            input_object = {
                "custom_id": resource.metadata.get("resource_id"),
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

    resources_tokens = total_tokens
    total_system_tokens = tokens_system_prompt * len(results)
    total_tokens.append(total_system_tokens)
    
    # Generate a unique 4-character ID
    unique_id = generate_unique_id()
    output_file = output_file.replace('.jsonl', f'_{unique_id}.jsonl')

    # Save to .jsonl file
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    total_input_tokens = np.sum(total_tokens)
    total_openai_queries = len(results)

    return total_input_tokens, total_openai_queries, resources_tokens


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


def measure_tokens_lenghts(file_path, tokens_lengths):
    plt.figure(figsize=(12, 3))
    plt.plot(tokens_lengths, marker='o')
    plt.title("Resources Tokens lengths")
    plt.ylabel("# tokens")
    plt.savefig(file_path)
    plt.close()