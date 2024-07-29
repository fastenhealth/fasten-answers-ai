import json
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings import OPENAI_MODEL_EMBEDDING



def get_total_tokens_from_string(string: str,
                                 encoding=OPENAI_MODEL_EMBEDDING) -> int:
    """Returns the number of tokens in a text string."""
    return len(encoding.encode(string))


def plot_lenghts(file_path,
                 measurements_list,
                 title,
                 y_label):
    plt.figure(figsize=(12, 3))
    plt.plot(measurements_list, marker='o')
    plt.title(title)
    plt.ylabel(y_label)
    plt.savefig(file_path)
    plt.close()


def create_json_from_chunks(chunks,
                            output_file: str,
                            output_plot_tokens:str,
                            output_plot_chars:str):
    json_data = {"entry": []}
    
    total_tokens_list = []
    total_chars_list = []

    for id, document in enumerate(chunks, start=1):
        text = document.text
        tokens = get_total_tokens_from_string(text)
        chars = len(document.text)
        
        json_data["entry"].append({
            "resource": text,
            "resourceType": document.metadata["resourceType"],
            "resource_id": document.metadata["resource_id"],
            "chunk_id": id,
            "chunk_size": chars,
            "total_tokens": tokens
        })
        
        total_tokens_list.append(tokens)
        total_chars_list.append(chars)
    
    with open(output_file, 'w') as outfile:
        json.dump(json_data, outfile, indent=2)
        
    plot_lenghts(output_plot_tokens, total_tokens_list,
                 title="Distribution of tokens per chunk",
                 y_label="# of tokens")
    plot_lenghts(output_plot_chars, total_chars_list,
                 title="Distribution of chars per chunk",
                 y_label="# of chars")
    
    return np.mean(total_tokens_list), np.mean(total_chars_list)
