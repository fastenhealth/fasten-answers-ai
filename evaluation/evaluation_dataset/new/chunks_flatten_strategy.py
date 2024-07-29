import os



from openai_api import generate_qa_file_batch_api, aprox_costs

from utils.create_chunks import create_chunks_from_flatten_text
from utils.fhir_flattener import flatten_bundle
from utils.helpers import load_params, create_flat_files_folder_if_not_exists


if __name__ == "__main__":
    params = load_params("./data/parameters.json")

    flatten_files_path = params["data"]["flat_files_folder_path"]
    fhir_chunks_file = params["data"]["fhir_chunks"]

    created_files = create_flat_files_folder_if_not_exists(flatten_files_path)

    # Create flatten resources and metrics
    sources_text_mean_lenght, \
        total_bundle_entries, \
        new_flatten_files = flatten_bundle(params["data"]["fhir_raw_file"], flatten_files_path)

    chunks, \
        total_chunks, \
        mean_tokens = create_chunks_from_flatten_text(created_files,
                                                      flatten_files_path,
                                                      fhir_chunks_file)

    total_input_tokens, \
        total_openai_queries = generate_qa_file_batch_api(resources=chunks,
                                                          output_file=output_openai_batch_requests_file)

    total_costs, \
        input_costs, \
        output_costs = aprox_costs(total_input_tokens, total_openai_queries)

    fhir_report = (
        f"Total bundle entries: {total_bundle_entries}\n"
        f"Mean chars per resource text: {sources_text_mean_lenght}\n"
        f"Total chunks: {total_chunks}\n"
        f"Total input tokens: {total_input_tokens}\n"
        f"Total openai queries: {total_openai_queries}\n"
        f"Total aproximate costs: {total_costs}\n"
        f"Total aproximate input costs: {input_costs}\n"
        f"Total aproximate output costs: {output_costs}\n"
    )

    with open(output_report_file, 'w') as output:
        output.write(fhir_report)
