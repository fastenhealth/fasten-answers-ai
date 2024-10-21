import csv
import os
import pandas as pd

from tqdm import tqdm

from app.config.settings import logger
from app.processor.files_processor import ensure_data_directory_exists, generate_output_filename


def generate_responses(params,
                       data: pd.DataFrame,
                       llm_client,
                       model_prompt,
                       batch_size=4):
    DEFAULT_PARAMS = {
        "n_predict": 400,
        "temperature": 0.0,
        "stop": ["<|end|>"],
        "repeat_last_n": 64,
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
        "stream": False,
    }
    DEFAULT_PARAMS["temperature"] = params["temperature"]
    DEFAULT_PARAMS["n_predict"] = params["tokens_to_predict"]

    # Verify if data directory exists
    data_dir = ensure_data_directory_exists("data_speed")

    output_file = os.path.join(data_dir, generate_output_filename(process="speed_generation", task="evaluation"))
    
    # Fieldnames for the CSV
    fieldnames = [
            "model",
            "context_size",
            "total_cores",
            "context",
            "question",
            "response",
            "temperature",
            "n_predict",
            "tokens_predicted",
            "tokens_evaluated",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
        ]

    with open(output_file, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()

        for _, row in tqdm(data.iterrows(), total=len(data), desc="Generating responses"):
            query = row["question"]
            context = row["context"]

            try:
                full_response = llm_client.chat(
                    query=query, context=context, stream=False, model_prompt=model_prompt, params=DEFAULT_PARAMS
                )
                
                result = {
                    "model": full_response["model"],
                    "context_size": row["context_size"],
                    "total_cores": params["total_cores"],
                    "context": context,
                    "question": query,
                    "response": full_response["content"],
                    "temperature": params["temperature"],
                    "n_predict": params["tokens_to_predict"],
                    "tokens_predicted": full_response["tokens_predicted"],
                    "tokens_evaluated": full_response["tokens_evaluated"],
                    "prompt_n": full_response["timings"]["prompt_n"],
                    "prompt_ms": full_response["timings"]["prompt_ms"],
                    "prompt_per_token_ms": full_response["timings"]["prompt_per_token_ms"],
                    "prompt_per_second": full_response["timings"]["prompt_per_second"],
                    "predicted_n": full_response["timings"]["predicted_n"],
                    "predicted_ms": full_response["timings"]["predicted_ms"],
                    "predicted_per_token_ms": full_response["timings"]["predicted_per_token_ms"],
                    "predicted_per_second": full_response["timings"]["predicted_per_second"],
                }

                dict_writer.writerow(result)
                output_file.flush()

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
