import json
import os

from evaluation.core.process_data.process_files import jsonl_dataset_to_dataframe
from evaluation.core.rag_requests.batch_requests import batch_requests


# Load config
with open(os.path.join(os.path.dirname(__file__), "/data/input_generate_responses.json"), "r") as config_file:
    config = json.load(config_file)

DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data"))
JSONL_FILE = os.path.join(DATA_DIR, "openai_outputs", config["JSONL_FILE"])
JSONL_OUTPUT_CSV = os.path.join(
    DATA_DIR, "openai_outputs", config["JSONL_FILE"].replace(".jsonl", ".csv"))
RAG_OUTPUT_CSV = os.path.join(
    DATA_DIR, "rag_generation", config["RAG_OUTPUT_CSV"])

SERVER_URL = config["SERVER_URL"]
CORES = config["CORES"]
CONTEXT_SIZE = config["CONTEXT_SIZE"]
TEXT_BOOST = config["TEXT_BOOST"]
EMBEDDING_BOOST = config["EMBEDDING_BOOST"]


def main():
    # Convert openai jsonl to dataframe
    df = jsonl_dataset_to_dataframe(JSONL_FILE, JSONL_OUTPUT_CSV)

    # Make sync batch requests to local llm
    batch_requests(
        server_url=SERVER_URL,
        input_file=df,
        question_column="openai_query",
        output_file=RAG_OUTPUT_CSV,
        optional_fields=["openai_answer", "resource_id_source"],
        extra_parameters={"cores": CORES, "context_size": CONTEXT_SIZE},
        text_boost=TEXT_BOOST,
        embedding_boost=EMBEDDING_BOOST,
    )


if __name__ == "__main__":
    main()
