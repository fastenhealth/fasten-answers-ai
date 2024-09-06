from app.config.settings import logger
from app.services.llama_client import llm_client


def summarize_resources(resources: list[dict],
                        stream: bool = False
                        ):
    for resource in resources:
        response = llm_client.chat(query=resource,
                                   stream=stream,
                                   task="summarize")
        summary = response.get("content")
        tokens_evaluated = response.get("tokens_evaluated")
        tokens_predicted = response.get("tokens_predicted")
        prompt_ms = response.get("prompt_ms")
        predicted_ms = response.get("predicted_ms")
        # Get values
        resource["summary"] = summary
        resource["tokens_evaluated"] = tokens_evaluated
        resource["tokens_predicted"] = tokens_predicted
        resource["prompt_ms"] = prompt_ms
        resource["predicted_ms"] = predicted_ms

        logger.info(
            f"This is the summary: {summary}")

    return resources
