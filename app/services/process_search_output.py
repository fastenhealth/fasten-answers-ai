from fastapi.responses import StreamingResponse

from .llama_client import llm_client
from ..config.settings import logger


def process_search_output(search_results):
    logger.info("Processing search results")
    processed_contents = []

    for result in search_results:
        content = result['content']
        metadata = result['metadata']

        # Create a title based on the metadata
        if metadata:
            title = ", ".join([f"{key}: {value}" for key,
                               value in metadata.items()])
            content_with_periods = content.replace('\n', '. ')
            processed_content = f"source: {title}\n{content_with_periods}"
        else:
            content_with_periods = content.replace('\n', '. ')
            processed_content = f"{content_with_periods}"

        processed_contents.append(processed_content)

    # Concatenate the processed contents
    concatenated_content = "\n\n".join(processed_contents)
    logger.info("Search results processed")
    return concatenated_content


def stream_llm_response(concatenated_content, query):
    logger.info("Streaming LLM response")

    def generate():
        for chunk in llm_client.chat(query, concatenated_content):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
