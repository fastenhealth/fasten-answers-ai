import json
from typing import List
import matplotlib.pyplot as plt

from llama_index.core.schema import BaseNode, Node
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sampling_data as sd
from settings import QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL, \
    OPENAI_MODEL, MAX_TOKENS
from sampling_data import get_total_tokens_from_string


def create_json_nodes(data: dict, text_splitter):
    nodes = []
    json_lengths_chars = []

    for entry in data.get("entry", []):
        if "resource" in entry:
            resource = entry["resource"]
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")

            node_text = json.dumps(resource)
            len_node_text = len(node_text)
            json_lengths_chars.append(len_node_text)

            chunks = text_splitter.split_text(node_text)
            for chunk in chunks:
                nodes.append(Node(
                    text=chunk.replace('\"', '').replace('\\', ''),  
                    metadata={"resourceType": resource_type,
                              "resource_id": resource_id}
                ))
    return nodes, json_lengths_chars


def create_json_from_nodes(nodes, output_file):
    json_data = {"entry": []}

    for id, node in enumerate(nodes, start=1):
        node.metadata["node_id"] = str(id)
        json_data["entry"].append({
            "resource": node.text,
            "resourceType": node.metadata["resourceType"],
            "resource_id": node.metadata["resource_id"],
            "node_id": node.metadata["node_id"]
        })
        

    with open(output_file, 'w') as outfile:
        json.dump(json_data, outfile, indent=2)
    
    return nodes


def create_plot_json_lenghts(file_path, section_lengths):
    plt.figure(figsize=(12, 3))
    plt.plot(section_lengths, marker='o')
    plt.title("Section lengths")
    plt.ylabel("# chars")
    plt.savefig(file_path)
    plt.close()


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
        custom_id = node.metadata["node_id"]
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

        total_tokens = get_total_tokens_from_string(system_prompt
                                                    + user_prompt)

        tokens_input.append(total_tokens)

    input_costs = sum(tokens_input) * (cost_per_million_input / 1000000)

    return round(output_costs + input_costs, 3), sum(tokens_input), tokens_outputs


answer_example = {
    'questions_and_answers': [
        {'question': 'What is renal dialysis?',
         'answer': 'Renal dialysis is a medical procedure used to remove waste products and excess fluid from the blood when the kidneys are not functioning properly. It is typically required for patients with end-stage renal disease.'},
        {'question': 'Why did I undergo renal dialysis?',
         'answer': 'You underwent renal dialysis due to end-stage renal disease, which is a condition where your kidneys have lost their ability to effectively filter blood, necessitating dialysis as a life-sustaining treatment.'},
        {'question': 'When was my renal dialysis performed?',
         'answer': 'Your renal dialysis was performed on September 21, 1996, starting at 12:25 PM and concluding at 4:03 PM.'},
        {'question': 'Where did my dialysis session take place?',
         'answer': 'Your dialysis session took place at the Worcester Outpatient Clinic.'},
        {'question': "What does the status 'completed' mean regarding my dialysis procedure?",
         'answer': "The status 'completed' indicates that the renal dialysis procedure was successfully performed and all necessary steps were carried out as intended, and you have completed that treatment session."}
    ]
}


if __name__ == "__main__":
    # Define Variables
    # ==============================================
    # Load the sample JSON data for a single patient
    with open("../data/Abe604_Runolfsdottir785_3718b84e-cbe9-1950-6c6c-e6f4fdc907be.json", "r") as f:
        json_data = json.load(f)
    # Get total tokens for system prompt
    system_prompt_tokens = sd.get_total_tokens_from_string(
        QUESTION_GEN_SYS_TMPL.format(num_questions_per_chunk=5))
    # Splitters size
    chunk_size = 600
    chunk_overlap = 50
    # ==============================================

    # Get a sample of max_entries from synthetic FHIR data with maximum 10k tokens each
    sampled_data, remaining_data = sd.sample_resources(json_data, system_prompt_tokens, max_entries=1000)

    # Save sample and remaining entries
    with open("../data/selected_entries.json", "w") as f:
        json.dump(sampled_data, f, indent=2)
    with open("../data/remaining_entries.json", "w") as f:
        json.dump(remaining_data, f, indent=2)

    # Define text splitter for JSON
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[',', ':', ' ', ''],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Create nodes and jsonl file for openai api batch call
    nodes, json_lenghts_chars = create_json_nodes(sampled_data, text_splitter)
    nodes = create_json_from_nodes(nodes, "../data/json_sample_parsed.json")
    create_plot_json_lenghts("../data/jsons_lenghts.png", json_lenghts_chars)
    generate_qa_file_batch_api(nodes=nodes, output_file="../data/batch_api_2.jsonl")
    # Approximate costs
    costs_batch_api, total_input_tokens, total_output_tokens = aprox_costs(nodes)
    print(f"Total costs batch api: {costs_batch_api}")
    print(f"Total aprox input tokens: {total_input_tokens}")
    print(f"Total aprox output tokens: {total_output_tokens}")
