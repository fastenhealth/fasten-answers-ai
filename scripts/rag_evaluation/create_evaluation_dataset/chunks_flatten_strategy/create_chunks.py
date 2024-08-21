import json
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings import OPENAI_MODEL_EMBEDDING


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


def get_total_tokens_from_string(string: str,
                                 encoding=OPENAI_MODEL_EMBEDDING) -> int:
    """Returns the number of tokens in a text string."""
    return len(encoding.encode(string))


def measure_tokens_lenghts(file_path, tokens_lengths):
    plt.figure(figsize=(12, 3))
    plt.plot(tokens_lengths, marker='o')
    plt.title("Tokens lengths")
    plt.ylabel("# tokens")
    plt.savefig(file_path)
    plt.close()


def create_chunks_from_text(text_files: List[str],
                            flat_file_path,
                            output_file: str,
                            text_splitter=text_splitter):
    documents = {}
    tokens_list = []
    chunk_counter = 1

    for file in text_files:
        with open(flat_file_path + "/" + file, 'r') as f:
            full_flatten_text = f.read()

        chunks = text_splitter.split_text(full_flatten_text)
        for chunk in chunks:
            key = f"{file[:-4]}_{chunk_counter}"
            documents[key] = {}

            documents[key]["file_name"] = file
            documents[key]["total_tokens"] = get_total_tokens_from_string(chunk)
            documents[key]["text_chunk"] = chunk

            tokens_list.append(documents[key]["total_tokens"])

            chunk_counter += 1

    with open(output_file, "w") as out:
        json.dump(documents, out, indent=2)

    measure_tokens_lenghts("../data/tokens_lenghts.png", tokens_list)
    return documents, chunk_counter, np.mean(tokens_list)
