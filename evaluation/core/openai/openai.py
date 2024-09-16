import requests

import tiktoken


def get_chat_completion(
    openai_api_key,
    user_prompt: str,
    system_prompt: str,
    answer_json_schema: dict,
    model: str,
    max_tokens: int = 300,
    endpoint_url: str = "https://api.openai.com/v1/chat/completions",
):
    """
    Function to get a chat completion from OpenAI's API using a specific system prompt and JSON schema.

    Parameters:
    - api_key: str, your OpenAI API key.
    - user_message: str, the user's message to the model.
    - system_prompt: str, the system prompt to guide the model's response.
    - model: str, the OpenAI model to use (default is "gpt-4o-2024-08-06").

    Returns:
    - dict, the response from the OpenAI API.
    """

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "response_format": answer_json_schema,
        "max_tokens": max_tokens,
    }

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


def get_total_tokens_from_string(string: str, encoding_name: str = "o200k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def calculate_total_tokens(
    df,
    query_column,
    generated_answer_column,
    contexts_column,
    reference_answer_column,
    system_template: str,
    user_template: str,
    template_type: str,
    encoding_name: str,
) -> int:
    """Calculates the total number of tokens required for the batch based on the type of evaluation."""
    total_tokens = 0

    for _, row in df.iterrows():
        if template_type == "correctness":
            user_prompt = user_template.format(
                query=row[query_column],
                reference_answer=row[reference_answer_column],
                generated_answer=row[generated_answer_column],
            )
        elif template_type == "faithfulness":
            user_prompt = user_template.format(generated_answer=row[generated_answer_column], contexts=row[contexts_column])

        system_prompt = system_template
        complete_prompt = system_prompt + user_prompt
        total_tokens += get_total_tokens_from_string(complete_prompt, encoding_name=encoding_name)

    return total_tokens


def calculate_api_costs(
    total_input_tokens: int,
    total_openai_requests: int,
    cost_per_million_input_tokens: float,
    cost_per_million_output_tokens: float,
    tokens_per_response: int,
) -> tuple:
    """
    Calculate the approximate costs of using an API, including the option to use Batch API pricing.

    Args:
        total_input_tokens (int): Total number of input tokens.
        total_openai_requests (int): Total number of OpenAI API queries.
        cost_per_million_input_tokens (float): Cost per million input tokens.
        cost_per_million_output_tokens (float): Cost per million output tokens.
        tokens_per_response (int): Number of tokens generated per API response.

    Returns:
        tuple: Total cost, input token cost, and output token cost.
    """

    # Calculate input token costs
    input_cost = round(total_input_tokens * (cost_per_million_input_tokens / 1_000_000), 3)

    # Calculate total output tokens and their costs
    total_output_tokens = total_openai_requests * tokens_per_response
    output_cost = round(total_output_tokens * (cost_per_million_output_tokens / 1_000_000), 3)

    # Calculate the total cost
    total_cost = round(input_cost + output_cost, 2)

    return {
        "total_cost": total_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
