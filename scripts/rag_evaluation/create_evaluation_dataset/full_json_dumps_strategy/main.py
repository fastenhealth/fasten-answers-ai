import numpy as np
import os

from create_chunks import create_json_from_chunks
from json_dumps import create_resource_documents, text_splitter, read_json_FHIR, \
    measure_texts_lengths
from openai_api import generate_qa_file_batch_api, aprox_costs, measure_tokens_lenghts
from settings import QUESTION_GEN_SYS_TMPL_1, QUESTION_GEN_SYS_TMPL_2, \
    create_directories
    

# Define paths for with and without URLs
path_with_urls = "./data/output/with_urls"
path_without_urls = "./data/output/no_urls"

# Parameters for experiment
chunk_size = 600
chunk_overlap = 50
data_strategy = "flatten or json dumps, no urls, etc"

remove_urls = False  # Change this to toggle between URLs and no URLs
fhir_file = './data/input/Abraham100_Oberbrunner298_9dbb826d-0be6-e8f9-3254-dbac25d83be6.json'

# Set the output folder depending on remove_urls
output_folder = path_without_urls if remove_urls is True else path_with_urls

flat_file_path = os.path.join(output_folder, 'flat_files')
output_json_chunks_file = os.path.join(output_folder, 'FHIR_chunks.json')
output_costs_report_file = 'costs_report'

output_plot_full_resource_chars_lenght = os.path.join(output_folder, "texts_resources_lengths.png") 
output_plot_chunks_tokens = os.path.join(output_folder, 'chunks_tokens_lenghts.png')
output_plot_chunks_chars = os.path.join(output_folder, 'chunks_chars_lenghts.png')



if __name__ == "__main__":
    # Create directories if not exists
    create_directories(path_with_urls, path_without_urls, remove_urls)

    # read FHIR file
    fhir_file = read_json_FHIR(fhir_file)

    # Create contexts
    documents_full, \
        documents_chunks, \
        json_lengths_chars = create_resource_documents(data=fhir_file,
                                                       remove_urls=remove_urls,
                                                       text_splitter=text_splitter(chunk_size, chunk_overlap))

    # Measure text lengths and save the image in the output folder
    measure_texts_lengths(output_plot_full_resource_chars_lenght,
                          json_lengths_chars)

    # Create json chunks file in the output folder
    mean_tokens_per_chunk, \
        mean_chars_per_chunk = create_json_from_chunks(documents_chunks,
                                                       output_json_chunks_file,
                                                       output_plot_chunks_tokens,
                                                       output_plot_chunks_chars)

    # Create Q/A with and without ID and dates
    system_prompts = {'ids_dates': QUESTION_GEN_SYS_TMPL_1,
                      'no_ids_dates': QUESTION_GEN_SYS_TMPL_2}

    for key, value in system_prompts.items():
        total_input_tokens, \
            total_openai_queries, \
            resources_tokens = generate_qa_file_batch_api(resources=documents_full,
                                                          system_prompt=value,
                                                          output_file=os.path.join(output_folder, f"openai_requests_{key}.jsonl"))
        total_costs, \
            input_costs, \
            output_costs = aprox_costs(total_input_tokens, total_openai_queries)

        # Measure token lengths and save the image in the output folder
        measure_tokens_lenghts(os.path.join(output_folder,
                                            f"{key}_input_tokens_openai_lengths.png"),
                               resources_tokens)

        # Create the costs report and save it in the output folder
        costs_report = (
            f"Total bundle entries: {len(documents_full)}\n"
            f"Mean chars per entry text: {np.mean(json_lengths_chars)}\n"
            f"Total chunks: {len(documents_chunks)}\n"
            f"Mean tokens per chunk: {mean_tokens_per_chunk}\n"
            f"Mean chars per chunk: {mean_chars_per_chunk}\n"
            f"Total input tokens: {total_input_tokens}\n"
            f"Total output tokens approximate: {total_openai_queries * 300}\n"
            f"Total openai queries: {total_openai_queries}\n"
            f"Total aproximate costs: {total_costs}\n"
            f"Total aproximate input costs: {input_costs}\n"
            f"Total aproximate output costs: {output_costs}\n"
        )

        with open(f"{output_folder}/{output_costs_report_file}_{key}.txt", 'w') as output:
            output.write(costs_report)
