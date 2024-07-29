import csv
import pandas as pd

from evaluation.core.rag_requests.rag_client import query_rag_server
import pdb


def batch_requests(server_url: str,
                   input_file: pd.DataFrame,
                   question_column: str,
                   output_file: str,
                   optional_fields: list[str] = None,
                   extra_parameters: dict = None,
                   text_boost: float = 1.0,
                   embedding_boost: float = 1.0):
    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            "query",
            "resources_id_contexts",
            "concatenated_contexts",
            "response",
            "tokens_predicted",
            "tokens_evaluated",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second"
        ]

        if optional_fields:
            fieldnames.extend(optional_fields)
        if extra_parameters:
            fieldnames.extend(extra_parameters.keys())

        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()

        for _, row in input_file.iterrows():
            question = row[question_column]

            rag_response = query_rag_server(server_url=server_url,
                                            query=question,
                                            k=5,
                                            threshold=0.2,
                                            stream=False,
                                            text_boost=text_boost,
                                            embedding_boost=embedding_boost).json()

            if optional_fields and rag_response:
                for field in optional_fields:
                    rag_response[field] = row[field]

            if extra_parameters and rag_response:
                for key, value in extra_parameters.items():
                    rag_response[key] = value

            dict_writer.writerow(rag_response)
            f.flush()
