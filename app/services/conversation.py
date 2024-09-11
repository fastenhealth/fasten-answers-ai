import time

from fastapi.responses import StreamingResponse

from app.services.llama_client import llm_client
from app.config.settings import logger, settings


def process_search_output(search_results):
    logger.info("Processing search results")
    processed_contents = []
    resources_id = []

    for result in search_results:
        content = result["content"]
        resource_id = result["metadata"]["resource_id"]

        processed_contents.append(content.replace("\\", ""))
        resources_id.append(resource_id)

    # Concatenate the processed contents
    concatenated_content = "\n\n".join(processed_contents)
    logger.info("Search results processed")
    logger.info(concatenated_content)
    return concatenated_content, resources_id


def llm_response(concatenated_context: str,
                 query: str,
                 resources_id: list,
                 stream: bool,
                 params: dict):
    if stream:

        def generate():
            start_time = time.time()
            for chunk in llm_client.chat(context=concatenated_context,
                                         query=query,
                                         stream=stream,
                                         params=params,
                                         model_prompt=settings.model.conversation_model_prompt):
                yield chunk
            elapsed_time = (time.time() - start_time) * 1000
            logger.info(
                f"stream_llm_response took {elapsed_time:.2f} milliseconds.")

        return StreamingResponse(generate(), media_type="text/plain")
    else:
        logger.info(stream)
        response = llm_client.chat(
            context=concatenated_context,
            query=query,
            stream=stream,
            params=params,
            model_prompt=settings.model.conversation_model_prompt)
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
