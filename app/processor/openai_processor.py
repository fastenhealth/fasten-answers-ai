import os
from datetime import datetime

from tqdm import tqdm

from services.openai import OpenAIHandler


def process_prompts_and_calculate_costs(
    system_prompt: str,
    user_prompts: list[dict],
    cost_per_million_input_tokens: float,
    cost_per_million_output_tokens: float,
    tokens_per_response: int,
    model: str = "gpt-4o-mini-2024-07-18"
) -> dict:
    """
    Process a list of user prompts, format the model prompt, calculate total tokens, and API costs.

    Args:
        user_prompts: list of dicts, each dict represents a user prompt.
        system_prompt: str, the system prompt.
        cost_per_million_input_tokens: float, cost per million input tokens.
        cost_per_million_output_tokens: float, cost per million output tokens.
        tokens_per_response: int, number of tokens generated per API response.
        model: str, the model to use.

    Returns:
        dict: Total cost and token information.
    """
    openai_handler = OpenAIHandler(model=model)

    total_input_tokens = 0
    total_openai_requests = len(user_prompts)

    for user_prompt in user_prompts:
        tokens_for_this_prompt = openai_handler.get_total_tokens_from_prompt(
            user_prompt)

        total_input_tokens += tokens_for_this_prompt

    total_system_tokens = openai_handler.get_total_tokens_from_prompt(
        system_prompt) * total_openai_requests

    total_input_tokens += total_system_tokens

    # Calculate api costs
    total_cost_info = openai_handler.calculate_api_costs(
        total_input_tokens=total_input_tokens,
        total_openai_requests=total_openai_requests,
        cost_per_million_input_tokens=cost_per_million_input_tokens,
        cost_per_million_output_tokens=cost_per_million_output_tokens,
        tokens_per_response=tokens_per_response
    )

    return total_cost_info


def ensure_data_directory_exists():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def generate_output_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"openai_responses_{timestamp}.csv"


def process_prompts_and_save_responses(
    system_prompt: str,
    user_prompts: list[dict],
    openai_api_key: str,
    model: str = "gpt-4o-mini-2024-07-18",
    max_tokens: int = 300
):
    """
    Process a list of user prompts, generate completions for each one, and save the results to a CSV file.

    Args:
        system_prompt: str, the system prompt.
        user_prompts: list of dicts, each dict represents a user prompt.
        openai_api_key: str, the OpenAI API key.
        model: str, the model to use.
        max_tokens: int, maximum number of tokens to generate per completion.

    Returns:
        None
    """
    openai_handler = OpenAIHandler(model=model)

    # Verify if data dir exists
    data_dir = ensure_data_directory_exists()

    # Generar el nombre del archivo de salida
    output_file = os.path.join(data_dir, generate_output_filename())

    # Determinar si el archivo ya existe para escribir el encabezado solo una vez
    file_exists = os.path.isfile(output_file)

    # Abrir el archivo una sola vez fuera del bucle
    with open(output_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['request', 'answer'])

        # Escribir el encabezado solo si el archivo no existe
        if not file_exists:
            writer.writeheader()

        # Iterar sobre cada prompt y obtener las respuestas de OpenAI
        for user_prompt in tqdm(user_prompts, desc="Generating completions"):
            user_content = user_prompt['content']
            request = f"System: {system_prompt}\nUser: {user_content}"

            # Generar la respuesta de OpenAI para el prompt
            completion = openai_handler.get_chat_completion(
                openai_api_key=openai_api_key,
                user_prompt=user_content,
                system_prompt=system_prompt,
                answer_json_schema={},  # Ajusta el esquema si es necesario
                max_tokens=max_tokens
            )

            if completion:
                answer = completion['choices'][0]['message']['content']
                writer.writerow({'request': request, 'answer': answer})

        # Al terminar, asegurarse de que todo se haya escrito al archivo
        file.flush()

    print(f"Responses saved to {output_file}")
