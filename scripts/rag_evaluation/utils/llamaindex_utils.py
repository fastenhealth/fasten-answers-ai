import numpy as np
import json
import pandas as pd
from requests.exceptions import HTTPError
from typing import Tuple, List, Dict

from llama_index.core.schema import BaseNode, Node
from llama_index.core import ChatPromptTemplate, PromptTemplate
import openai

from .sampling_data import get_total_tokens_from_string
from .settings import QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL, \
    OPENAI_MODEL, OPENAI_MODEL_EMBEDDING, MAX_TOKENS, logger


def create_json_nodes_llamaindex_test(data: dict):
    nodes = []
    for entry in data.get("entry", []):
        if "resource" in entry:
            resource = entry["resource"]
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            node_text = json.dumps(resource)
            nodes.append(Node(
                text=node_text,
                metadata={"resourceType": resource_type, "id": resource_id}
            ))
    return nodes


def generate_nodes_tokens_stats(
    nodes: List[BaseNode],
    question_gen_template: ChatPromptTemplate,
    encoding=OPENAI_MODEL_EMBEDDING,
    num_questions_per_chunk: int = 10
) -> List[Tuple[str, str]]:
    """Generate tokens stats."""
    tokens_stats = []

    for _, node in enumerate(nodes):
        context_str = node.get_content(metadata_mode="all")
        fmt_messages = question_gen_template.format_messages(
            num_questions_per_chunk=num_questions_per_chunk,
            context_str=context_str,
        )
        text = fmt_messages[0].content + " " + fmt_messages[1].content

        num_tokens = get_total_tokens_from_string(text, encoding)
        tokens_stats.append(num_tokens)

    return tokens_stats


def generate_tokens_percentiles_table(data):
    percentiles = np.arange(0, 101, 10)
    percentile_values = np.percentile(data, percentiles)
    df = pd.DataFrame(percentile_values, index=[f'{p}%' for p in percentiles], columns=['Value'])
    return df


def openai_create_completion(system_prompt,
                             user_prompt,
                             max_tokens=MAX_TOKENS,
                             OpenAI_model=OPENAI_MODEL):
    try:
        response = openai.chat.completions.create(
            model=OpenAI_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens
        )
        return response
    except HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred: {err}")
    return None


def generate_qa_pairs(
    context_str: str,
    system_prompt=QUESTION_GEN_SYS_TMPL,
    user_prompt=QUESTION_GEN_USER_TMPL,
    num_questions_per_chunk: int = 5
) -> Dict[str, Dict[str, str]]:
    """Generate questions."""

    system_prompt = system_prompt.format(
        num_questions_per_chunk=num_questions_per_chunk)
    user_prompt = user_prompt.format(
        context_str=context_str
    )

    total_tokens = get_total_tokens_from_string(system_prompt
                                                + user_prompt)
    if total_tokens < 10000:
        openai_response = openai_create_completion(system_prompt,
                                                   user_prompt)
        results = {
            "context": context_str,
            "openai_response": openai_response
        }
    else:
        openai_response = {"context": f"Total tokens {total_tokens} exceded limit",
                           "result": 0}
    return results
