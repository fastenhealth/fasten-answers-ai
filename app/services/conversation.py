import csv
import os
import pandas as pd
import time
from tqdm import tqdm

from fastapi.responses import StreamingResponse

from app.services.llama_client import llm_client
from app.config.settings import logger, settings
from app.data_models.search_result import SearchResult
from app.processor.files_processor import ensure_data_directory_exists, generate_output_filename
from app.services.search_documents import search_query


def process_search_output(search_results: list[SearchResult]):
    processed_contents = []
    resources_id = []

    for result in search_results:
        content = result.content
        resource_id = result.metadata["resource_id"]

        processed_contents.append(content.replace("\\", ""))
        resources_id.append(resource_id)

    # Concatenate the processed contents
    concatenated_content = "\n\n".join(processed_contents)

    return concatenated_content, resources_id


def llm_response(concatenated_context: str, query: str, resources_id: list, stream: bool, params: dict):
    if stream:

        def generate():
            start_time = time.time()
            for chunk in llm_client.chat(
                context=concatenated_context,
                query=query,
                stream=stream,
                params=params,
                model_prompt=settings.model.conversation_model_prompt,
            ):
                yield chunk
            elapsed_time = (time.time() - start_time) * 1000
            logger.info(f"stream_llm_response took {elapsed_time:.2f} milliseconds.")

        return StreamingResponse(generate(), media_type="text/plain")
    else:
        logger.info(stream)
        response = llm_client.chat(
            context=concatenated_context,
            query=query,
            stream=stream,
            params=params,
            model_prompt=settings.model.conversation_model_prompt,
        )
        logger.info(f"Response received: {response}")
        result = {
            "query": query,
            "resources_id_contexts": resources_id,
            "concatenated_contexts": concatenated_context,
            "response": response["content"],
            "tokens_predicted": response["tokens_predicted"],
            "tokens_evaluated": response["tokens_evaluated"],
            "prompt_n": response["timings"]["prompt_n"],
            "prompt_ms": response["timings"]["prompt_ms"],
            "prompt_per_token_ms": response["timings"]["prompt_per_token_ms"],
            "prompt_per_second": response["timings"]["prompt_per_second"],
            "predicted_n": response["timings"]["predicted_n"],
            "predicted_ms": response["timings"]["predicted_ms"],
            "predicted_per_token_ms": response["timings"]["predicted_per_token_ms"],
            "predicted_per_second": response["timings"]["predicted_per_second"],
        }
        return result


def batch_generation_synchronous(
    model_prompt: str,
    es_client,
    embedding_model,
    input_data: pd.DataFrame,
    question_column: str,
    k: int,
    text_boost: float,
    embedding_boost: float,
    llm_model: str,
    process: str = "local_llm_response",
    job: str = "generation_evaluation",
) -> str:
    """
    Batch generation, saves results to a CSV file, and loads summaries into Elasticsearch.
    """
    # Verify if data directory exists
    data_dir = ensure_data_directory_exists()

    # Output filename
    output_file = os.path.join(data_dir, generate_output_filename(process=process, task=job))

    # Fieldnames for the CSV
    fieldnames = [
        "resource_id_source",
        "openai_query",
        "openai_answer",
        "context",
        "resources_id_context",
        "local_llm_model",
        "model_prompt",
        "response",
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

    # Open CSV file to write results
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in tqdm(input_data.iterrows(), total=len(input_data), desc="Generating responses"):
            user_query = row[question_column]
            # Query
            context = search_query(
                query_text=user_query,
                embedding_model=embedding_model,
                es_client=es_client,
                k=k,
                text_boost=text_boost,
                embedding_boost=embedding_boost,
            )
            if not context:
                concatenated_context = "There is no context"
            else:
                concatenated_context, resources_id = process_search_output(context)
            # Get answer
            try:
                response = llm_client.chat(
                    query=user_query, stream=False, model_prompt=model_prompt, context=concatenated_context
                )

                if not response or not isinstance(response, dict):
                    logger.error(f"Invalid response for query: {user_query}")
                    continue

                rag_response = {
                    "resource_id_source": row["resource_id_source"],
                    "openai_query": user_query,
                    "openai_answer": row["openai_answer"],
                    "context": concatenated_context,
                    "resources_id_context": resources_id,
                    "local_llm_model": llm_model,
                    "model_prompt": model_prompt,
                    "response": response["content"],
                    "tokens_predicted": response["tokens_predicted"],
                    "tokens_evaluated": response["tokens_evaluated"],
                    "prompt_n": response["timings"]["prompt_n"],
                    "prompt_ms": response["timings"]["prompt_ms"],
                    "prompt_per_token_ms": response["timings"]["prompt_per_token_ms"],
                    "prompt_per_second": response["timings"]["prompt_per_second"],
                    "predicted_n": response["timings"]["predicted_n"],
                    "predicted_ms": response["timings"]["predicted_ms"],
                    "predicted_per_token_ms": response["timings"]["predicted_per_token_ms"],
                    "predicted_per_second": response["timings"]["predicted_per_second"],
                }

                writer.writerow(rag_response)

                # Flush the file after each batch
                file.flush()

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
    return output_file
