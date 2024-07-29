import json
from typing import Tuple, List, Dict

from llama_index.core.schema import BaseNode, Node

from .settings import QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL, \
    OPENAI_MODEL, OPENAI_MODEL_EMBEDDING, MAX_TOKENS, logger
from .sampling_data import get_total_tokens_from_string


def create_json_nodes_llamaindex_batch_api(data: dict):
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


def generate_qa_file_batch_api(
    nodes: List[BaseNode],
    system_prompt=QUESTION_GEN_SYS_TMPL,
    user_prompt=QUESTION_GEN_USER_TMPL,
    max_tokens=MAX_TOKENS,
    num_questions_per_chunk: int = 5,
    output_file: str = "qa_file.jsonl"
) -> None:
    """Generate questions and save to a .jsonl file."""
    results = []
    
    method = "POST"
    url = "/v1/chat/completions"
    system_prompt = system_prompt.format(
        num_questions_per_chunk=num_questions_per_chunk)
    
    for node in nodes:
        custom_id = node.metadata["id"]        
        user_prompt_formatted = user_prompt.format(
            context_str=node.get_content(metadata_mode="all")
        )
        input_object = {
            "custom_id": custom_id,
            "method": method,
            "url": url,
            "body": {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_formatted}
                ],
                "max_tokens": max_tokens
            }
        }
        results.append(input_object)
    
    # Save to .jsonl file
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
            

def aprox_costs( 
    nodes: List[BaseNode],
    system_prompt=QUESTION_GEN_SYS_TMPL,
    user_prompt=QUESTION_GEN_USER_TMPL,
    num_questions_per_chunk: int = 5,
    cost_per_million_input: float = 0.075,
    cost_per_million_output: float = 0.3, 
    tokens_generated: int = 300
) -> None:
    """Generate aprox costs of using API"""
    tokens_input = []
    input_costs = []
    tokens_outputs = sum([tokens_generated] * len(nodes))
    output_costs = tokens_outputs * (cost_per_million_output / 1000000)

    system_prompt = system_prompt.format(
        num_questions_per_chunk=num_questions_per_chunk)
    
    for node in nodes:
        
        user_prompt = node.get_content()
        
        total_tokens = get_total_tokens_from_string(system_prompt \
            + user_prompt)
        
        tokens_input.append(total_tokens)
    
    input_costs = sum(tokens_input) * (cost_per_million_input / 1000000)
        
    return round(output_costs + input_costs, 3), sum(tokens_input), tokens_outputs



