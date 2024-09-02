import csv
from tqdm import tqdm


def generate_responses(params, contexts, questions, llm_client, batch_size=8):
    output_file_name = params["output_file"]

    with open(f"./data/{output_file_name}", "w", newline="") as output_file:
        fieldnames = [
            "model",
            "context_size",
            "total_cores",
            "prompt",
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

        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()

        for i in tqdm(range(0, len(contexts), batch_size)):
            batch_questions = [questions[context_size] for context_size in list(contexts.keys())[i : i + batch_size]]
            context_texts = [contexts[context_size] for context_size in list(contexts.keys())[i : i + batch_size]]
            responses = llm_client.chat(user_prompts=context_texts, questions=batch_questions)
            for i, full_response in enumerate(responses):
                context_size = list(contexts.keys())[i]
                context = context_texts[i]
                question = batch_questions[i]

                result = {
                    "model": full_response["model"],
                    "context_size": context_size,
                    "total_cores": params["total_cores"],
                    "prompt": context,
                    "question": question,
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
