import csv
import os

from elasticsearch import helpers
import traceback

from app.config.settings import logger, settings
from app.db.index_documents import bulk_load_fhir_data
from app.processor.files_processor import ensure_data_directory_exists, \
    generate_output_filename
from app.services.llama_client import llm_client


def summarize_resources(resources: list[dict], stream: bool = False):
    """
    Summarizes resources by calling the LLM client for each resource.
    """
    for resource in resources:
        try:
            response = llm_client.chat(
                query=resource,
                stream=stream,
                task="summarize",
                model_prompt=settings.model.summaries_model_prompt
            )
            summary = response.get("content")
            resource.update({
                "summary": summary,
                "tokens_evaluated": response.get("tokens_evaluated"),
                "tokens_predicted": response.get("tokens_predicted"),
                "prompt_ms": response.get("prompt_ms"),
                "predicted_ms": response.get("predicted_ms")
            })
            logger.info(
                f"Resource summarized: {resource['resource_id']}, Summary: {summary}")
        except Exception as e:
            logger.error(
                f"Error processing batch: {str(e)}\n{traceback.format_exc()}")
    return resources


async def summarize_resources_parallel(model_prompt: str,
                                       es_client,
                                       embedding_model,
                                       resources: list[dict],
                                       batch_size: int = 4) -> str:
    """
    Summarizes resources in parallel, saves results to a CSV file, and loads summaries into Elasticsearch.
    """
    # Verify if data directory exists
    data_dir = ensure_data_directory_exists()

    # Output filename
    output_file = os.path.join(data_dir, generate_output_filename(
        process="local_llm_response", task="summarize"))

    # Fieldnames for the CSV
    fieldnames = [
        "resource_id", "resource", "resource_type", "summary",
        "tokens_predicted", "tokens_evaluated", "prompt_n", "prompt_ms",
        "prompt_per_token_ms", "prompt_per_second", "predicted_n",
        "predicted_ms", "predicted_per_token_ms", "predicted_per_second"
    ]

    final_results = []

    # Open CSV file to write results
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(resources), batch_size):
            resource_batch = resources[i:i + batch_size]
            try:
                # Process the batch in parallel using the LLM client
                result = await llm_client.process_parallel(
                    resource_batch=resource_batch, model_prompt=model_prompt)

                extracted_results = []
                for resource, response in zip(resource_batch, result):
                    if not response or not isinstance(response, dict):
                        logger.error(
                            f"Invalid response for resource {resource['resource_id']}")
                        continue
                    extracted_response = {
                        "resource_id": resource["resource_id"],
                        "resource": resource["resource"],
                        "resource_type": resource["resource_type"],
                        "summary": response["content"],
                        "tokens_predicted": response["tokens_predicted"],
                        "tokens_evaluated": response["tokens_evaluated"],
                        "prompt_n": response["timings"]["prompt_n"],
                        "prompt_ms": response["timings"]["prompt_ms"],
                        "prompt_per_token_ms": response["timings"]["prompt_per_token_ms"],
                        "prompt_per_second": response["timings"]["prompt_per_second"],
                        "predicted_n": response["timings"]["predicted_n"],
                        "predicted_ms": response["timings"]["predicted_ms"],
                        "predicted_per_token_ms": response["timings"]["predicted_per_token_ms"],
                        "predicted_per_second": response["timings"]["predicted_per_second"]
                    }

                    writer.writerow(extracted_response)
                    extracted_results.append(extracted_response)
                    final_results.append(extracted_response)

                    # Flush the file after each batch
                    file.flush()

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
    # Load results into Elasticsearch
    helpers.bulk(
        es_client,
        bulk_load_fhir_data(
            data=final_results,
            text_key="summary",
            embedding_model=embedding_model,
            index_name=settings.elasticsearch.index_name,
        )
    )

    return output_file
