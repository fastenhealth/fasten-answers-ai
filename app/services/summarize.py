from tqdm import tqdm
from app.config.settings import logger
from app.services.llama_client import llm_client
from app.config.settings import settings


async def summarize_resources(resources: list[dict],
                        stream: bool = False,
                        batch_size: int = 4
                        ):
    for i in tqdm(range(0, len(resources), batch_size)):
    # for resource in tqdm(resources):
        batch_resources = resources[i:i + batch_size]
        contexts = [None for _ in range(len(batch_resources))]
        responses = await llm_client.chat_parallel(settings.model.summaries_model_prompt, queries=batch_resources, contexts=contexts)
        for resource, response in zip(batch_resources, responses):
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
