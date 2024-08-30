import requests


def get_chat_completion(
    openai_api_key,
    user_prompt: str,
    system_prompt: str,
    answer_json_schema: dict,
    model: str,
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
