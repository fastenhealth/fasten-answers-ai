from collections import defaultdict
import json
import random

from llama_index.core.schema import BaseNode, Node

from .settings import OPENAI_MODEL_EMBEDDING


import json
import random
from typing import List


def get_total_tokens_from_string(string: str,
                                 encoding=OPENAI_MODEL_EMBEDDING) -> int:
    """Returns the number of tokens in a text string."""
    return len(encoding.encode(string))


def sample_resources(data: dict, system_prompt_tokens: int, max_entries=1000):
    # Step 1: Count resource types and collect entries
    resource_type_count = {}
    resource_entries = {}
    for entry in data.get("entry", []):
        if "resource" in entry:
            resource = entry["resource"]
            resource_type = resource.get("resourceType")
            total_tokens = get_total_tokens_from_string(json.dumps(resource)) \
                + system_prompt_tokens
            if total_tokens > 10000:
                continue  # Exclude entries with more than 10000 tokens
            if resource_type:
                if resource_type not in resource_type_count:
                    resource_type_count[resource_type] = 0
                    resource_entries[resource_type] = []
                resource_type_count[resource_type] += 1
                resource_entries[resource_type].append(entry)
    
    # Step 2: Sample entries
    sampled_entries = []
    for resource_type, entries in resource_entries.items():
        if len(entries) <= 20:
            sampled_entries.extend(entries)
        else:
            sampled_entries.extend(random.sample(entries, 20))
    
    # Adjust the number of entries to max_entries
    additional_entries_needed = max_entries - len(sampled_entries)
    
    if additional_entries_needed > 0:
        remaining_pool = [entry for entries in resource_entries.values() for entry in entries if entry not in sampled_entries]
        if len(remaining_pool) > additional_entries_needed:
            sampled_entries.extend(random.sample(remaining_pool, additional_entries_needed))
        else:
            sampled_entries.extend(remaining_pool)
    
    # Ensure sampled entries are not more than max_entries
    if len(sampled_entries) > max_entries:
        sampled_entries = sampled_entries[:max_entries]
    
    # Separate the remaining entries
    sampled_ids = {entry["resource"]["id"] for entry in sampled_entries}
    remaining_entries = [entry for entry in data.get("entry", []) if entry["resource"]["id"] not in sampled_ids]
    
    # Create the final JSON structures
    sampled_data = {key: data[key] for key in data if key != "entry"}
    sampled_data["entry"] = sampled_entries
    
    remaining_data = {key: data[key] for key in data if key != "entry"}
    remaining_data["entry"] = remaining_entries
    
    return sampled_data, remaining_data


def create_fhir_json(selected_nodes, remaining_nodes):
    selected_fhir = {
        "entry": [node["entry"] for node in selected_nodes]
    }
    
    remaining_fhir = {
        "entry": [node["entry"] for node in remaining_nodes]
    }

    return selected_fhir, remaining_fhir


def sample_one_per_resource_type(selected_fhir):
    resource_type_dict = defaultdict(list)
    for entry in selected_fhir["entry"]:
        resource_type = entry["resource"]["resourceType"]
        resource_type_dict[resource_type].append(entry)

    selected_entries = []
    for resource_type, entries in resource_type_dict.items():
        selected_entry = random.choice(entries)
        selected_entries.append(selected_entry)
    
    new_selected_fhir = {
        "entry": selected_entries
    }
    
    return new_selected_fhir
