import os

from fhir_flattener import flatten_bundle
from create_chunks import create_chunks_from_text
from openai_api import generate_qa_file_batch_api, aprox_costs


fhir_file = "../data/Abraham100_Oberbrunner298_9dbb826d-0be6-e8f9-3254-dbac25d83be6.json"
flat_file_path = "../data/flat_files"

output_json_file = "../data/FHIR_chunks.json"
output_report_file = "../data/FHIR_report.txt"
output_openai_batch_requests_file = "../data/openai_requests.jsonl"


if __name__ == "__main__":
    if not os.path.exists(flat_file_path):
        os.mkdir(flat_file_path)
    # Create flatten resources and metrics
    sources_text_mean_lenght, total_bundle_entries = flatten_bundle(fhir_file, flat_file_path)

    created_files = os.listdir(flat_file_path)

    chunks, total_chunks, mean_tokens = create_chunks_from_text(created_files, flat_file_path, output_json_file)

    total_input_tokens, total_openai_queries = generate_qa_file_batch_api(
        resources=chunks, output_file=output_openai_batch_requests_file
    )

    total_costs, input_costs, output_costs = aprox_costs(total_input_tokens, total_openai_queries)

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

    with open(output_report_file, "w") as output:
        output.write(fhir_report)
